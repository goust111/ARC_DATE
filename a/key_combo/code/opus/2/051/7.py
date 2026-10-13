from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dr, dc in _DIRS:
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _perimeter(bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    ps = []
    for cc in range(c0, c1 + 1):
        ps.append((r0, cc))
        ps.append((r1, cc))
    for rr in range(r0 + 1, r1):
        ps.append((rr, c0))
        ps.append((rr, c1))
    return ps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for comp in _components(g == 5):
        if len(comp) < 8:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        perim = set(_perimeter((r0, c0, r1, c1)))
        comp_set = set(comp)
        miss = list(perim - comp_set)
        if len(miss) != 1:
            continue
        gap = miss[0]
        gap_side: str
        if gap[0] == r0:
            gap_side = "top"
        elif gap[0] == r1:
            gap_side = "bottom"
        elif gap[1] == c0:
            gap_side = "left"
        else:
            gap_side = "right"

        # Inner seeds (non-gray pixels)
        seed_positions = []
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                v = int(g[rr, cc])
                if v != 0 and v != 5:
                    seed_positions.append((rr, cc, v))
        if len(seed_positions) < 2:
            continue
        # Need two seeds same row/col.
        color_a, color_b = seed_positions[0][2], seed_positions[1][2]
        if seed_positions[0][0] != seed_positions[1][0] and seed_positions[0][1] != seed_positions[1][1]:
            continue
        seeds = seed_positions[:2]
        p1, p2 = seeds[0], seeds[1]
        same_row = p1[0] == p2[0]

        # Hat direction from missing-neighbor outside the perimeter.
        hat_cells = []
        for dr, dc in _DIRS:
            nr, nc = gap[0] + dr, gap[1] + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 5 and (nr, nc) not in comp_set:
                hat_cells.append((nr, nc))
        if not hat_cells:
            if gap_side in ("top", "bottom"):
                hat_cells.append((gap[0], gap[1] - 1))
            else:
                hat_cells.append((gap[0] - 1, gap[1]))

        # Stripe axis from gap side.
        if gap_side in ("top", "bottom"):
            axis = "v"
            d = abs(p2[0] - p1[0])
            start_side = "left"
            if hat_cells[0][1] > gap[1]:
                start_side = "right"
        else:
            axis = "h"
            d = abs(p2[1] - p1[1])
            start_side = "top"
            if hat_cells[0][0] > gap[0]:
                start_side = "bottom"
        if d <= 0:
            d = 1
        c1, c2 = p1[2], p2[2]
        if start_side in ("left", "top"):
            first_color, second_color = c1, c2
        else:
            first_color, second_color = c2, c1

        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1 - 0 + 1 - 1):
                if axis == "v":
                    idx = (cc - c0 - 1) // d
                    if start_side == "right":
                        idx = (c1 - 1 - cc) // d
                else:
                    idx = (rr - r0 - 1) // d
                    if start_side == "bottom":
                        idx = (r1 - 1 - rr) // d
                if idx % 2 == 0:
                    out[rr, cc] = first_color
                else:
                    out[rr, cc] = second_color

    return out


def generate_input() -> np.ndarray:
    h = 18
    w = 18
    g = np.zeros((h, w), dtype=int)
    colors = [1, 2, 3, 4, 6, 7]
    random.shuffle(colors)
    used: set[Tuple[int, int]] = set()
    for fi in range(random.randint(2, 3)):
        color_a, color_b = colors[2 * fi % len(colors)], colors[(2 * fi + 1) % len(colors)]
        hi = random.randint(6, 8)
        wi = random.randint(6, 8)
        if hi < 4 or wi < 4:
            continue
        for _ in range(200):
            r0 = random.randint(1, h - hi - 2)
            c0 = random.randint(1, w - wi - 2)
            r1 = r0 + hi - 1
            c1 = c0 + wi - 1
            if any((r, c) in used for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
                continue
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                gap_col = random.randint(c0 + 1, c1 - 1)
                gap = (r0, gap_col)
                hat = [(r0 - 1, gap_col), (r0 - 1, gap_col + 1 if gap_col + 1 <= c1 else gap_col - 1)]
            elif side == "bottom":
                gap_col = random.randint(c0 + 1, c1 - 1)
                gap = (r1, gap_col)
                hat = [(r1 + 1, gap_col), (r1 + 1, gap_col + 1 if gap_col + 1 <= c1 else gap_col - 1)]
            elif side == "left":
                gap_row = random.randint(r0 + 1, r1 - 1)
                gap = (gap_row, c0)
                hat = [(gap_row, c0 - 1), (gap_row + 1 if gap_row + 1 <= r1 else gap_row - 1, c0 - 1)]
            else:
                gap_row = random.randint(r0 + 1, r1 - 1)
                gap = (gap_row, c1)
                hat = [(gap_row, c1 + 1), (gap_row + 1 if gap_row + 1 <= r1 else gap_row - 1, c1 + 1)]

            # Draw frame border minus gap.
            for cc in range(c0, c1 + 1):
                if (r0, cc) != gap:
                    g[r0, cc] = 5
                if (r1, cc) != gap:
                    g[r1, cc] = 5
            for rr in range(r0 + 1, r1):
                if (rr, c0) != gap:
                    g[rr, c0] = 5
                if (rr, c1) != gap:
                    g[rr, c1] = 5
            for rr, cc in hat:
                if 0 <= rr < h and 0 <= cc < w:
                    g[rr, cc] = 5

            # Two seed pixels in same row or column.
            if random.random() < 0.5:
                seed_r = random.randint(r0 + 1, r1 - 1)
                c_a = random.randint(c0 + 1, c1 - 3)
                g[seed_r, c_a] = color_a
                g[seed_r, c_a + random.randint(1, 3)] = color_b
            else:
                seed_c = random.randint(c0 + 1, c1 - 1)
                r_a = random.randint(r0 + 1, r1 - 3)
                g[r_a, seed_c] = color_a
                g[r_a + random.randint(1, 3), seed_c] = color_b

            used.update((rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1))
            break
    return g
