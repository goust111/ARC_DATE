import random
from collections import deque
from typing import List, Tuple

import numpy as np


def _cross_lines(g: np.ndarray):
    rows = np.where(np.all(g == 2, axis=1))[0]
    cols = np.where(np.all(g == 2, axis=0))[0]
    H, W = g.shape
    r0 = int(rows[0]) if len(rows) else H // 2
    c0 = int(cols[0]) if len(cols) else W // 2
    return r0, c0


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    visited = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or visited[r, c]:
                continue
            q = [(r, c)]
            visited[r, c] = True
            comp = [(r, c)]
            touches = False
            while q:
                x, y = q.pop()
                if x == 0 or y == 0 or x == h - 1 or y == w - 1:
                    touches = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if nx < 0 or nx >= h or ny < 0 or ny >= w:
                        continue
                    if not bg[nx, ny] or visited[nx, ny]:
                        continue
                    visited[nx, ny] = True
                    q.append((nx, ny))
                    comp.append((nx, ny))
            if not touches:
                holes += 1
    return holes


def _fill_holes(local_img: np.ndarray) -> Tuple[np.ndarray, int]:
    h, w = local_img.shape
    mask = (local_img != 0)
    holes = _count_holes(mask)
    if holes == 0:
        return (np.where(mask, 8, 0), holes)

    out = local_img.copy()
    bg = ~mask
    visited = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if not bg[r, c] or visited[r, c]:
                continue
            q = [(r, c)]
            visited[r, c] = True
            comp = [(r, c)]
            touches = False
            while q:
                x, y = q.pop()
                if x == 0 or y == 0 or x == h - 1 or y == w - 1:
                    touches = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if nx < 0 or nx >= h or ny < 0 or ny >= w:
                        continue
                    if not bg[nx, ny] or visited[nx, ny]:
                        continue
                    visited[nx, ny] = True
                    q.append((nx, ny))
                    comp.append((nx, ny))
            if not touches:
                for hx, hy in comp:
                    out[hx, hy] = 6
    return out, holes


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    cr, cc = _cross_lines(g)

    panels = [
        (0, cr, 0, cc),
        (0, cr, cc + 1, W),
        (cr + 1, H, 0, cc),
        (cr + 1, H, cc + 1, W),
    ]
    out = np.zeros((cr, cc), dtype=int)

    layers = []
    for ridx, (r0, r1, c0, c1) in enumerate(panels):
        sub = g[r0:r1, c0:c1]
        if sub.size == 0:
            continue
        mask = (sub != 0) & (sub != 2)
        if not np.any(mask):
            continue
        rs = np.where(mask.any(axis=1))[0]
        cs = np.where(mask.any(axis=0))[0]
        if rs.size == 0 or cs.size == 0:
            continue
        rr0, rr1 = rs[0], rs[-1]
        cc0, cc1 = cs[0], cs[-1]
        local = sub[rr0:rr1 + 1, cc0:cc1 + 1].copy()
        local[~((local != 0) & (local != 2))] = 0

        filled, holes = _fill_holes(local)

        # map offset in output canvas
        off_r = 0 if r0 == 0 else 0
        if r0 == 0:
            off_r = 0
        else:
            off_r = - (cr + 1)
        off_c = 0 if c0 == 0 else 0 if c0 == cc + 1 else 0

        # better direct mapping by quadrant
        if r0 == 0 and c0 == 0:
            canvas_r0, canvas_c0 = 0, 0
        elif r0 == 0 and c0 == cc + 1:
            canvas_r0, canvas_c0 = 0, 0
        elif r0 == cr + 1 and c0 == 0:
            canvas_r0, canvas_c0 = -1, 0
            # will use rr0+(-1) == rr0-cr-1? handled below
        else:
            canvas_r0, canvas_c0 = -1, 0

        # easier: compute within panel local dimensions and place preserving relative coordinates
        if r0 == 0 and c0 == 0:
            qr = rr0
            qc = cc0
        elif r0 == 0 and c0 == cc + 1:
            qr = rr0
            qc = cc0 - (cc + 1)
        elif r0 == cr + 1 and c0 == 0:
            qr = rr0 - (cr + 1)
            qc = cc0
        else:
            qr = rr0 - (cr + 1)
            qc = cc0 - (cc + 1)

        layers.append((holes, ridx, qr, qc, filled))

    for _, _, roff, coff, layer in sorted(layers):
        for lr in range(layer.shape[0]):
            for lc in range(layer.shape[1]):
                v = int(layer[lr, lc])
                if v == 0:
                    continue
                rr = roff + lr
                cc2 = coff + lc
                if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
                    out[rr, cc2] = v
    return out


def _templates(h: int = 1):
    # returns masks with increasing hole count
    return [
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=int,
        ),
    ]


def generate_input() -> np.ndarray:
    H = 15
    W = 15
    g = np.zeros((H, W), dtype=int)
    cr = H // 2
    cc = W // 2
    g[cr, :] = 2
    g[:, cc] = 2

    colors = random.sample([1, 3, 4, 7, 8], 4)
    masks = _templates()
    targets = [0, 1, 2, 3]

    # panel bounds
    panels = [
        (0, cr, 0, cc),
        (0, cr, cc + 1, W),
        (cr + 1, H, 0, cc),
        (cr + 1, H, cc + 1, W),
    ]

    for (r0, r1, c0, c1), color, target in zip(panels, colors, targets):
        mask = masks[min(target, len(masks) - 1)]
        ph, pw = mask.shape
        max_r = r1 - r0
        max_c = c1 - c0
        rr0 = r0 + random.randint(0, max(0, max_r - ph - 1))
        cc0 = c0 + random.randint(0, max(0, max_c - pw - 1))
        for dr in range(ph):
            for dc in range(pw):
                if mask[dr, dc]:
                    g[rr0 + dr, cc0 + dc] = color

    return g
