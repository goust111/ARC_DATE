from __future__ import annotations

import random

import numpy as np


def _find_legend(g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 2):
            a = int(g[r, c])
            b = int(g[r, c + 1])
            if a == 0 or b == 0 or a == b:
                continue
            neighborhood: list[int] = []
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 3):
                    if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != 0:
                        neighborhood.append((rr, cc))
            if set(neighborhood) == {(r, c), (r, c + 1)}:
                candidates.append((r, c))
    if len(candidates) != 1:
        raise ValueError("expected exactly one isolated horizontal two-color legend")
    return candidates[0]


def _find_marker(g: np.ndarray, legend_cells: set[tuple[int, int]]) -> tuple[int, int]:
    h, w = g.shape
    markers: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if (r, c) in legend_cells:
                continue
            if int(g[r, c]) != 0:
                markers.append((r, c))
    if len(markers) != 1:
        raise ValueError("expected exactly one non-background boundary marker")
    return markers[0]


def _inward_direction(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    if c == w - 1:
        return 0, -1
    raise ValueError("marker is not on an edge")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw an inward ray whose pass-through wall color and rewrite color are encoded by an isolated legend.

    Input structure:
    - A black canvas has one non-black marker pixel on the outer border.
    - Interior one-cell-thick wall segments use exactly two wall colors.
    - One isolated horizontal two-pixel legend has a left cell matching the pass-through wall color and a right cell with the rewrite color.

    Rule:
    - Erase the two legend cells.
    - Starting next to the boundary marker, advance straight inward.
    - Paint black cells with the marker color, recolor pass-through wall cells to the legend's right color, and stop before any other non-black cell.

    Key insight:
    The legend is not a static color replacement table; it changes which wall color the ray can cross and how crossed wall cells are rewritten.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    lr, lc = _find_legend(g)
    pass_color = int(g[lr, lc])
    rewrite_color = int(g[lr, lc + 1])
    legend_cells = {(lr, lc), (lr, lc + 1)}
    marker = _find_marker(g, legend_cells)
    marker_color = int(g[marker])
    out[lr, lc] = 0
    out[lr, lc + 1] = 0

    h, w = g.shape
    dr, dc = _inward_direction(marker[0], marker[1], h, w)
    r, c = marker[0] + dr, marker[1] + dc
    while 0 <= r < h and 0 <= c < w:
        value = int(g[r, c])
        if (r, c) in legend_cells:
            value = 0
        if value == 0:
            out[r, c] = marker_color
        elif value == pass_color:
            out[r, c] = rewrite_color
        else:
            break
        r += dr
        c += dc
    return out


def _clear_near_legend(cells: list[tuple[int, int]]) -> bool:
    return all(max(abs(r - 1), abs(c - 1)) > 1 and max(abs(r - 1), abs(c - 2)) > 1 for r, c in cells)


def _add_segment(g: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    h, w = g.shape
    if not all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in cells):
        return False
    if not _clear_near_legend(cells):
        return False
    if any(int(g[r, c]) != 0 for r, c in cells):
        return False
    for r, c in cells:
        g[r, c] = color
    return True


def generate_input() -> np.ndarray:
    for _attempt in range(200):
        h = random.randint(10, 14)
        w = random.randint(10, 14)
        g = np.zeros((h, w), dtype=int)
        marker_color, pass_color, block_color, rewrite_color = random.sample(range(1, 10), 4)
        g[1, 1] = pass_color
        g[1, 2] = rewrite_color

        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            mr, mc, dr, dc = 0, random.randint(4, w - 2), 1, 0
            max_step = h - 2
        elif side == "bottom":
            mr, mc, dr, dc = h - 1, random.randint(4, w - 2), -1, 0
            max_step = h - 2
        elif side == "left":
            mr, mc, dr, dc = random.randint(4, h - 2), 0, 0, 1
            max_step = w - 2
        else:
            mr, mc, dr, dc = random.randint(4, h - 2), w - 1, 0, -1
            max_step = w - 2
        g[mr, mc] = marker_color

        if max_step < 6:
            continue
        pass_step = random.randint(2, max_step - 3)
        block_step = random.randint(pass_step + 2, max_step)
        crossing_specs = [(pass_step, pass_color), (block_step, block_color)]
        ok = True
        for step, color in crossing_specs:
            rr = mr + dr * step
            cc = mc + dc * step
            length = random.randint(3, 6)
            half = length // 2
            if dr != 0:
                start = max(1, min(w - length - 1, cc - half))
                cells = [(rr, c) for c in range(start, start + length)]
            else:
                start = max(1, min(h - length - 1, rr - half))
                cells = [(r, cc) for r in range(start, start + length)]
            ok = _add_segment(g, cells, color)
            if not ok:
                break
        if not ok:
            continue

        extra_count = random.randint(1, 3)
        for _ in range(extra_count):
            color = random.choice([pass_color, block_color])
            length = random.randint(3, 6)
            for _try in range(40):
                horizontal = random.choice([True, False])
                if horizontal:
                    rr = random.randint(2, h - 2)
                    cc = random.randint(1, w - length - 1)
                    cells = [(rr, c) for c in range(cc, cc + length)]
                else:
                    rr = random.randint(2, h - length - 1)
                    cc = random.randint(1, w - 2)
                    cells = [(r, cc) for r in range(rr, rr + length)]
                if _add_segment(g, cells, color):
                    break
        return g
    raise RuntimeError("failed to generate puzzle input")
