from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

if str(Path('/home/lixinyu/snap/new/solvers')) not in sys.path:
    sys.path.append(str(Path('/home/lixinyu/snap/new/solvers')))

from arc_utils import Color

BLACK = Color.BLACK
YELLOW = Color.YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """Detect large solid color rectangles as obstacles and trace a deflecting 2-pixel yellow diagonal band.

    Input structure: noisy grid with two hidden large same-color rectangles and one unique yellow source at (h-1,0).

    Rule: detect maximal rectangles (side>=4) and treat them as obstacles, trace a two-cell-wide upward-right diagonal band from
    the source first row above, shifting right until each step is obstacle-free, and deflecting instantly right if collision.
    Keep obstacle colors and band pixels, clear all others to black.

    Key insight: obstacle detection is a perception step; only after finding large rectangles can the band deflection path be computed.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects 2D ndarray")

    h, w = g.shape
    obstacles = _find_max_rectangles(g)

    srcs = np.argwhere((g == int(YELLOW)))
    if srcs.size == 0:
        return np.zeros_like(g)
    sr, sc = map(int, srcs[0])

    obstacle_set = set()
    for r0, c0, r1, c1, _col in obstacles:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                obstacle_set.add((r, c))

    out = np.zeros_like(g)

    # keep all obstacle colors
    for r0, c0, r1, c1, col in obstacles:
        out[r0 : r1 + 1, c0 : c1 + 1] = int(col)

    # source stays yellow
    out[sr, sc] = int(YELLOW)

    r = sr - 1
    left = sc
    right = sc + 1
    prev_left = left
    prev_right = right

    while r >= 0:
        if left >= w and right >= w:
            break

        # deflect if obstacle overlap
        while (0 <= left < w and (r, left) in obstacle_set) or (0 <= right < w and (r, right) in obstacle_set) or left < 0 or right < 0:
            left += 1
            right += 1
        if left >= w or right >= w:
            break

        # draw band
        if 0 <= left < w:
            out[r, left] = int(YELLOW)
        if 0 <= right < w:
            out[r, right] = int(YELLOW)

        # fill connection gap for continuity
        if r < sr - 1:
            for c in range(min(prev_left, prev_right) + 1, max(prev_left, prev_right) + 1):
                if c >= left and c <= right:
                    continue
        if r + 1 >= 0:
            if left > prev_right + 1:
                for c in range(prev_right + 1, left):
                    if 0 <= c < w:
                        out[r + 1, c] = int(YELLOW)

        prev_left, prev_right = left, right
        r -= 1
        left += 1
        right += 1

    return out


def _find_max_rectangles(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = g.shape
    colors = sorted({int(v) for v in g.flatten()})
    candidates: list[tuple[int, int, int, int, int]] = []
    for color in colors:
        if color == int(BLACK):
            continue
        m = (g == color).astype(np.uint8)
        for r0 in range(h):
            for c0 in range(w):
                for r1 in range(r0 + 3, h):
                    for c1 in range(c0 + 3, w):
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        if area < 16:
                            continue
                        sub = m[r0 : r1 + 1, c0 : c1 + 1]
                        if int(sub.sum()) != area:
                            continue
                        # check if maximal (not strictly contained by a larger rect same color)
                        is_max = True
                        for rr0 in range(max(0, r0 - 1), r0 + 1):
                            for cc0 in range(max(0, c0 - 1), c0 + 1):
                                for rr1 in range(r1, min(h - 1, r1 + 1) + 1):
                                    for cc1 in range(c1, min(w - 1, c1 + 1) + 1):
                                        if rr0 == r0 and cc0 == c0 and rr1 == r1 and cc1 == c1:
                                            continue
                                        if rr0 <= r0 and cc0 <= c0 and rr1 >= r1 and cc1 >= c1:
                                            if int(m[rr0 : rr1 + 1, cc0 : cc1 + 1].sum()) == (rr1 - rr0 + 1) * (cc1 - cc0 + 1):
                                                is_max = False
                        if not is_max:
                            continue
                        candidates.append((r0, c0, r1, c1, color))

    # deduplicate by exact tuple, keep biggest coverage order
    unique = []
    seen = set()
    for c in sorted(candidates, key=lambda t: (t[4], (t[2] - t[0] + 1) * (t[3] - t[1] + 1), t[0], t[1])):
        key = (c[0], c[1], c[2], c[3], c[4])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # choose outermost rectangles only
    return unique


def generate_input() -> np.ndarray:
    h, w = 16, 16
    g = np.full((h, w), int(BLACK), dtype=int)

    # two hidden large rectangles
    colors = [Color.RED, Color.BLUE]
    rects = [
        (2, 2, 5, 6, int(colors[0])),
        (7, 9, 11, 12, int(colors[1])),
    ]
    for r0, c0, r1, c1, color in rects:
        g[r0 : r1 + 1, c0 : c1 + 1] = color

    # random noise
    palette = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    for _ in range(40):
        r = random.randint(0, h - 1)
        c = random.randint(0, w - 1)
        if g[r, c] == int(BLACK):
            if random.random() < 0.7:
                g[r, c] = int(random.choice(palette))

    # source at bottom-left
    g[h - 1, 0] = int(YELLOW)
    # ensure no extra yellow
    g[(g == int(YELLOW)) & np.array(np.indices((h, w))[0] < h - 1 | (np.indices((h, w))[1] != 0))] = int(BLACK)

    return g
