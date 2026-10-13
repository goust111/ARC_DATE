from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc051_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


def _rects(g):
    rects = []
    for col in set(np.unique(g).tolist()) - {0, 3, 7}:
        for r in U.find_hollow_rects_by_color(g, col):
            rects.append((col, r))
    return rects


def _cross_side(g):
    comp = max(U.components(g == 7), key=len)
    s = set(comp)
    center = None
    for p in comp:
        if sum((q in s) for q in U.n4(*p)) >= 4:
            center = p
            break
    if center is None:
        r0, c0, r1, c1 = U.bbox(comp)
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
    r, c = center
    dirs = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
    lengths = {}
    for name, (dr, dc) in dirs.items():
        n = 0
        x, y = r + dr, c + dc
        while (x, y) in s:
            n += 1
            x += dr
            y += dc
        lengths[name] = n
    longest = max(lengths, key=lengths.get)
    return {"up": "right", "right": "down", "down": "left", "left": "up"}[longest]


def main(grid: np.ndarray) -> np.ndarray:
    """Keep the largest framed sprite and attach its mirror on the marker-selected side.

    Input structure:
    Several colored hollow frames contain green sprites. An orange cross has one
    arm longer than the other three.

    Rule:
    Select the frame with largest strict inner area, erase all other frames and
    contents, erase the selected frame border and the cross, and keep only its
    green sprite. The longest cross arm rotated clockwise chooses the side where
    a mirrored copy of that sprite is placed flush against the original.

    Key insight:
    Area comparison chooses the surviving object, while an independent marker
    supplies the mirror direction.
    """
    g = np.array(grid, copy=True)
    out = np.zeros_like(g)
    rects = _rects(g)
    _, win = max(rects, key=lambda x: (x[1][2] - x[1][0] - 1) * (x[1][3] - x[1][1] - 1))
    r0, c0, r1, c1 = win
    inside = np.zeros_like(g, dtype=bool)
    inside[r0 + 1:r1, c0 + 1:c1] = True
    greens = [(r, c) for r, c in zip(*np.where((g == 3) & inside))]
    for r, c in greens:
        out[r, c] = 3
    sr0, sc0, sr1, sc1 = U.bbox(greens)
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    side = _cross_side(g)
    for r, c in greens:
        rr, cc = r - sr0, c - sc0
        if side in ("left", "right"):
            nr = sr0 + rr
            nc = (sc1 + 1 if side == "right" else sc0 - sw) + (sw - 1 - cc)
        else:
            nr = (sr1 + 1 if side == "down" else sr0 - sh) + (sh - 1 - rr)
            nc = sc0 + cc
        if U.in_bounds(out.shape, nr, nc):
            out[nr, nc] = 3
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((16, 16), dtype=int)
    rects = [(1, (1, 1, 6, 5)), (2, (1, 9, 7, 14)), (4, (9, 2, 14, 8))]
    random.shuffle(rects)
    for col, rect in rects[:random.randint(2, 3)]:
        U.draw_rect_border(g, *rect, col)
        r0, c0, r1, c1 = rect
        for _ in range(random.randint(3, 5)):
            r = random.randint(r0 + 1, r1 - 1)
            c = random.randint(c0 + 1, c1 - 1)
            g[r, c] = 3
    cr, cc = 10, 12
    lens = {"up": 1, "right": 1, "down": 1, "left": 1}
    long = random.choice(list(lens))
    lens[long] = random.randint(2, 3)
    g[cr, cc] = 7
    for name, (dr, dc) in {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}.items():
        for k in range(1, lens[name] + 1):
            g[cr + dr * k, cc + dc * k] = 7
    return g
