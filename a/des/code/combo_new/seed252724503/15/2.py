import random
from collections import deque

import numpy as np


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _normalize(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    return {(r - r0, c - c0) for r, c in cells}


def _connected(cells):
    cells = list(cells)
    if not cells:
        return False
    cell_set = set(cells)
    q = deque([cells[0]])
    seen = {cells[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if n in cell_set and n not in seen:
                seen.add(n)
                q.append(n)
    return len(seen) == len(cell_set)


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touch = False
            while q:
                rr, cc = q.popleft()
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touch = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touch:
                holes += 1
    return holes


def _reconstruct(a_cells, b_cells):
    a = list(a_cells)
    b = list(b_cells)
    best = None
    for dr in range(-25, 26):
        for dc in range(-25, 26):
            moved = {(r + dr, c + dc) for r, c in b}
            union = set(a) | moved
            if len(union) != len(a) + len(b):
                continue
            if not _connected(union):
                continue
            rs = [r for r, _ in union]
            cs = [c for _, c in union]
            area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            edge = 0
            for r, c in a:
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (r + dr2, c + dc2) in moved:
                        edge += 1
            score = (area, -edge)
            if best is None or score < best[0]:
                best = (score, union)
    return best[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each same-color fragment pair into a single completed shape, count its holes, and pack the completed shapes left-to-right by increasing hole count.

    Input structure:
    - The grid has a dominant background color.
    - Each foreground color appears in exactly two disconnected monochrome fragments.
    - For each color, the two fragments can be joined by translation without rotation to form one connected finished shape.

    Rule:
    - For each color, find the translation of one fragment that produces a connected non-overlapping union with the other fragment.
    - Count the number of enclosed background holes in that finished union.
    - Normalize each finished shape to its tight bounding box.
    - Pack the shapes in ascending hole-count order, top-aligned with one background column between consecutive shapes.

    Key insight:
    - The sorting key is invisible until after the fragments are reassembled, so reconstruction must happen before the hole-based ordering can be read.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    finished = []
    for color in [int(v) for v in np.unique(g) if v != bg]:
        comps = _components(g == color)
        union = _reconstruct(comps[0], comps[1])
        norm = _normalize(union)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        finished.append((_hole_count(mask), color, mask))
    finished.sort(key=lambda x: x[0])
    out_h = max(mask.shape[0] for _, _, mask in finished)
    out_w = sum(mask.shape[1] for _, _, mask in finished) + len(finished) - 1
    out = np.full((out_h, out_w), bg, dtype=int)
    c0 = 0
    for _, color, mask in finished:
        h, w = mask.shape
        out[:h, c0 : c0 + w][mask] = color
        c0 += w + 1
    return out


def _split_shape(mask: np.ndarray):
    h, w = mask.shape
    if random.choice([True, False]):
        seam = []
        x = random.randint(1, w - 1)
        for _ in range(h):
            x = min(w - 1, max(1, x + random.choice([-1, 0, 1])))
            seam.append(x)
        a = np.zeros_like(mask)
        b = np.zeros_like(mask)
        for r in range(h):
            a[r, : seam[r]] = mask[r, : seam[r]]
            b[r, seam[r] :] = mask[r, seam[r] :]
    else:
        seam = []
        x = random.randint(1, h - 1)
        for _ in range(w):
            x = min(h - 1, max(1, x + random.choice([-1, 0, 1])))
            seam.append(x)
        a = np.zeros_like(mask)
        b = np.zeros_like(mask)
        for c in range(w):
            a[: seam[c], c] = mask[: seam[c], c]
            b[seam[c] :, c] = mask[seam[c] :, c]
    return a, b


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(18, 28), random.randint(18, 28)
        bg = 0
        grid = np.full((h, w), bg, dtype=int)
        occ = np.zeros((h, w), dtype=bool)
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 3))
        ok = True
        for idx, color in enumerate(colors):
            sh, sw = random.randint(5, 7), random.randint(5, 7)
            mask = np.ones((sh, sw), dtype=bool)
            for _ in range(idx):
                rr = random.randint(1, sh - 2)
                cc = random.randint(1, sw - 2)
                mask[rr, cc] = False
            a, b = _split_shape(mask)
            comps = [np.argwhere(part) for part in (a, b)]
            if any(len(comp) == 0 for comp in comps):
                ok = False
                break
            parts = []
            for comp in comps:
                r0, c0 = comp.min(axis=0)
                r1, c1 = comp.max(axis=0)
                part = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
                for r, c in comp:
                    part[r - r0, c - c0] = True
                if not _connected({tuple(x) for x in np.argwhere(part)}):
                    ok = False
                    break
                parts.append(part)
            if not ok:
                break
            placed = 0
            for part in parts:
                ph, pw = part.shape
                done = False
                for _ in range(100):
                    top = random.randint(1, h - ph - 1)
                    left = random.randint(1, w - pw - 1)
                    if occ[max(0, top - 1) : min(h, top + ph + 1), max(0, left - 1) : min(w, left + pw + 1)].any():
                        continue
                    for r in range(ph):
                        for c in range(pw):
                            if part[r, c]:
                                grid[top + r, left + c] = color
                    occ[max(0, top - 1) : min(h, top + ph + 1), max(0, left - 1) : min(w, left + pw + 1)] = True
                    done = True
                    placed += 1
                    break
                if not done:
                    ok = False
                    break
            if not ok or placed != 2:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
