from __future__ import annotations

import random

import numpy as np
from arc_utils import find_components


GREY = 5
BG = 0


def _side_color(g: np.ndarray, side: str) -> int:
    if side == "top":
        vals = g[0, 1:-1]
    elif side == "bottom":
        vals = g[-1, 1:-1]
    elif side == "left":
        vals = g[1:-1, 0]
    else:
        vals = g[1:-1, -1]
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey fence interior with the color indicated by the fence gap's edge direction.

    Input structure:
    - Black background with one-pixel-wide colored bands on the four outer grid edges.
    - Several grey rectangular fences lie away from the border; each fence has exactly one one-pixel gap on one side.
    - Fence interiors are black.

    Rule:
    - Detect each fence and identify which side contains its gap.
    - Read the color of the matching outer edge band (top, bottom, left, or right).
    - Fill every black interior pixel of that fence with the matched edge color.
    - Keep the grey borders, edge bands, and outside background unchanged.

    Key insight:
    - The gap direction does not encode the fill color directly; it indexes the corresponding colored border legend.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    side_to_color = {side: _side_color(g, side) for side in ("top", "bottom", "left", "right")}
    for comp in find_components(g, bg=BG, colors=[GREY], connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == g.shape[0] - 1 or c1 == g.shape[1] - 1:
            continue
        top = int(np.sum(g[r0, c0 : c1 + 1] == GREY))
        bottom = int(np.sum(g[r1, c0 : c1 + 1] == GREY))
        left = int(np.sum(g[r0 : r1 + 1, c0] == GREY))
        right = int(np.sum(g[r0 : r1 + 1, c1] == GREY))
        side = "top"
        if top < (c1 - c0 + 1):
            side = "top"
        elif bottom < (c1 - c0 + 1):
            side = "bottom"
        elif left < (r1 - r0 + 1):
            side = "left"
        elif right < (r1 - r0 + 1):
            side = "right"
        fill = side_to_color[side]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner == BG] = fill
    return out


def _draw_frame(g: np.ndarray, r0: int, c0: int, h: int, w: int, side: str) -> None:
    r1, c1 = r0 + h - 1, c0 + w - 1
    g[r0, c0 : c1 + 1] = GREY
    g[r1, c0 : c1 + 1] = GREY
    g[r0 : r1 + 1, c0] = GREY
    g[r0 : r1 + 1, c1] = GREY
    if side in {"top", "bottom"}:
        c = random.randint(c0 + 1, c1 - 1)
        g[r0 if side == "top" else r1, c] = BG
    else:
        r = random.randint(r0 + 1, r1 - 1)
        g[r, c0 if side == "left" else c1] = BG


def generate_input() -> np.ndarray:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    g = np.zeros((h, w), dtype=int)
    band_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    g[0, :] = band_colors[0]
    g[-1, :] = band_colors[1]
    g[:, 0] = band_colors[2]
    g[:, -1] = band_colors[3]
    n = random.randint(1, 3)
    used: list[tuple[int, int, int, int]] = []
    for _ in range(n):
        for _ in range(100):
            fh = random.randint(4, 8)
            fw = random.randint(4, 8)
            r0 = random.randint(2, h - fh - 3)
            c0 = random.randint(2, w - fw - 3)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if all(box[2] < ur0 - 1 or box[0] > ur1 + 1 or box[3] < uc0 - 1 or box[1] > uc1 + 1 for ur0, uc0, ur1, uc1 in used):
                used.append(box)
                _draw_frame(g, r0, c0, fh, fw, random.choice(["top", "bottom", "left", "right"]))
                break
    return g
