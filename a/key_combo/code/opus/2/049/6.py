from __future__ import annotations

import random
from collections import deque
import numpy as np

BG = 0
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
PURPLE = 7
MAGENTA = 6

CORNER_MARKERS = [RED, BLUE, GREEN, PURPLE]
TEMPLATES = {
    (0, 0): [(1, 1), (1, 2), (2, 1), (2, 2)],
    (0, 7): [(1, 5), (1, 6), (2, 5), (2, 6)],
    (7, 0): [(6, 1), (5, 1), (6, 2)],
    (7, 7): [(6, 5), (5, 6), (6, 6)],
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build corner templates from corner markers and reflect upper-half magenta upward detail onto the lower half with template blocking.

    Input structure:
    - 8x8 black background.
    - 2~3 corners contain marker colors (red/blue/green/orange).
    - Upper half contains a small magenta connected component.

    Rule:
    - Remove each corner marker, draw its fixed template at the fixed corner coordinates with marker color.
    - Reflect the upper half vertically to the lower half (row i -> H-1-i).
    - A reflected magenta cell writes only if target is black; otherwise it is blocked by a template pixel.
    - Keep the original upper half unchanged.

    Key insight:
    - Corner markers are instructions for opaque barriers; reflection fills are conditioned on those barriers, so the same mirrored map yields different outputs.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    corners = {(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)}
    for r, c in corners:
        val = int(out[r, c])
        if val in CORNER_MARKERS:
            out[r, c] = BG
            template_cells = TEMPLATES[(r, c)]
            for tr, tc in template_cells:
                if 0 <= tr < H and 0 <= tc < W:
                    out[tr, tc] = val

    half = H // 2
    for r in range(half):
        rr = H - 1 - r
        for c in range(W):
            if out[r, c] != MAGENTA:
                continue
            if out[rr, c] == BG:
                out[rr, c] = MAGENTA

    return out


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0 in range(h):
        for c0 in range(w):
            if not bool(mask[r0, c0]) or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def generate_input() -> np.ndarray:
    for _ in range(1200):
        H = W = 8
        g = np.full((H, W), BG, dtype=int)

        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        marker_count = random.randint(2, 3)
        chosen = random.sample(corners, marker_count)
        marker_colors = random.sample(CORNER_MARKERS, marker_count)
        for (r, c), col in zip(chosen, marker_colors):
            g[r, c] = col

        # Build upper magenta component in upper half not touching edges.
        component_size = random.randint(3, 6)
        attempts = 0
        while attempts < 300:
            attempts += 1
            r0 = random.randint(1, H // 2 - 1)
            c0 = random.randint(1, W - 2)
            if any(g[r0 + 1, c0 + 1 : c0 + 3] == MAGENTA):
                pass
            used = {(r0, c0)}
            growth = [(r0, c0)]
            while len(growth) < component_size:
                frontier = []
                for rr, cc in growth:
                    for nr, nc in _neighbors4(rr, cc):
                        if 0 <= nr < H // 2 and 0 <= nc < W and g[nr, nc] == BG and (nr, nc) not in used:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                growth.append(random.choice(frontier))
                used.add(growth[-1])

            if len(growth) < component_size:
                continue

            valid = True
            for rr, cc in growth:
                if (rr, cc) in chosen:
                    valid = False
                    break
            if not valid:
                continue

            for rr, cc in growth:
                g[rr, cc] = MAGENTA

            # Ensure reflective constraints: at least one suppressed and one not suppressed.
            out = main(g)
            reflected = []
            for rr, cc in growth:
                rr2 = H - 1 - rr
                reflected.append((rr2, cc))

            blocked = sum(1 for rr, cc in reflected if g[rr, cc] != BG or out[rr, cc] == MAGENTA)
            free = sum(1 for rr, cc in reflected if out[rr, cc] == MAGENTA and g[rr, cc] == BG)
            if blocked == 0 or free == 0:
                g = np.full((H, W), BG, dtype=int)
                for (r, c), col in zip(chosen, marker_colors):
                    g[r, c] = col
                continue

            return g

    raise RuntimeError("failed to generate valid puzzle 6 input")

