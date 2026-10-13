import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marked loose piece into the matching notch of the grey template.

    Input structure:
    - Black background with one large grey template component.
    - Single colored anchor pixels sit in black notches along the template.
    - Away from the template are small multicolor pieces, each containing one anchor color.

    Rule:
    - Pair each loose piece with the template anchor of the same color.
    - Translate the whole piece so its own anchor pixel coincides with the template anchor.
    - Clear the piece's old location to black and paste it into the notch, preserving anchor colors.

    Key insight:
    - The colored anchor is both the matching key and the exact alignment point for the translation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    grey_pts = [(r, c) for r, c in zip(*np.where(g == GREY))]
    if not grey_pts:
        return out
    gr = [p[0] for p in grey_pts]
    gc = [p[1] for p in grey_pts]
    tr0, tr1 = min(gr), max(gr)
    tc0, tc1 = min(gc), max(gc)

    fg_mask = (g != BG) & (g != GREY)
    template_anchors = {}
    pieces = []
    for pts in _components(fg_mask):
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        inside_template_box = min(rows) >= tr0 and max(rows) <= tr1 and min(cols) >= tc0 and max(cols) <= tc1
        if len(pts) == 1 and inside_template_box:
            p = pts[0]
            template_anchors[int(g[p])] = p
        else:
            pieces.append(pts)

    for pts in pieces:
        anchor = None
        for p in pts:
            color = int(g[p])
            if color in template_anchors:
                anchor = p
                target = template_anchors[color]
                break
        if anchor is None:
            continue
        dr = target[0] - anchor[0]
        dc = target[1] - anchor[1]
        for r, c in pts:
            out[r, c] = BG
        for r, c in pts:
            nr, nc = r + dr, c + dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = int(g[r, c])
    return out


SHAPES = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
]


def _can_place(cells, occupied, h, w, margin=0):
    for r, c in cells:
        if not (margin <= r < h - margin and margin <= c < w - margin):
            return False
        if (r, c) in occupied:
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        h = rng.randint(18, 30)
        w = rng.randint(18, 30)
        g = np.zeros((h, w), dtype=int)
        th = rng.randint(8, min(14, h - 4))
        tw = rng.randint(8, min(14, w - 4))
        tr0 = rng.randint(2, h - th - 2)
        tc0 = rng.randint(2, w - tw - 2)
        tr1, tc1 = tr0 + th - 1, tc0 + tw - 1
        g[tr0 : tr1 + 1, tc0 : tc1 + 1] = GREY
        n = rng.randint(2, 4)
        if 2 * n > len(colors):
            continue
        anchor_colors = rng.sample(colors, n)
        fill_colors = rng.sample([c for c in colors if c not in anchor_colors], n)
        occupied = {(r, c) for r in range(tr0, tr1 + 1) for c in range(tc0, tc1 + 1)}
        used_template = set()
        ok = True
        for acol, fcol in zip(anchor_colors, fill_colors):
            shape = rng.choice(SHAPES)
            placed = False
            for _try in range(100):
                side = rng.choice(["top", "bottom", "left", "right"])
                sr = max(r for r, _c in shape) + 1
                sc = max(c for _r, c in shape) + 1
                if side == "top":
                    ar = tr0 + rng.randint(1, min(2, th - sr - 1))
                    ac = rng.randint(tc0 + 1, tc1 - sc)
                elif side == "bottom":
                    ar = tr1 - rng.randint(1, min(2, th - sr - 1)) - sr + 1
                    ac = rng.randint(tc0 + 1, tc1 - sc)
                elif side == "left":
                    ar = rng.randint(tr0 + 1, tr1 - sr)
                    ac = tc0 + rng.randint(1, min(2, tw - sc - 1))
                else:
                    ar = rng.randint(tr0 + 1, tr1 - sr)
                    ac = tc1 - rng.randint(1, min(2, tw - sc - 1)) - sc + 1
                cells = {(ar + r, ac + c) for r, c in shape}
                if cells & used_template:
                    continue
                for r, c in cells:
                    g[r, c] = BG
                g[ar, ac] = acol
                used_template |= cells
                placed = True
                break
            if not placed:
                ok = False
                break
            piece_cells_rel = shape
            piece_h = max(r for r, _c in piece_cells_rel) + 1
            piece_w = max(c for _r, c in piece_cells_rel) + 1
            placed_piece = False
            for _try in range(200):
                pr = rng.randint(1, h - piece_h - 1)
                pc = rng.randint(1, w - piece_w - 1)
                cells = {(pr + r, pc + c) for r, c in piece_cells_rel}
                if any(tr0 - 1 <= r <= tr1 + 1 and tc0 - 1 <= c <= tc1 + 1 for r, c in cells):
                    continue
                if not _can_place(cells, occupied, h, w):
                    continue
                occupied |= cells
                for r, c in cells:
                    g[r, c] = fcol
                g[pr, pc] = acol
                placed_piece = True
                break
            if not placed_piece:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("generate_input failed")
