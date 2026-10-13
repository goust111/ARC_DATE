from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

if str(Path('/home/lixinyu/snap/new/solvers')) not in sys.path:
    sys.path.append(str(Path('/home/lixinyu/snap/new/solvers')))

from arc_utils import Color

BLACK = Color.BLACK
GRAY = Color.GREY
BLUE = Color.BLUE
RED = Color.RED


def main(grid: np.ndarray) -> np.ndarray:
    """Decode missing-mask directions from upper-diff patterns and cast red rays in the lower half.

    Input structure: a gray divider row separates two aligned upper patterns (same bbox), one complete and one with black holes,
    with a single red source below and several 2x2 blue obstacle blocks.

    Rule: align the two upper patterns, extract missing-mask coordinates, normalize each displacement from the mask center
    to a direction, cast red rays from the source in those directions until a blue obstacle or boundary stop, then erase upper
    area and divider.

    Key insight: the upper difference mask is not output content but an encoded set of ray directions.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects 2D ndarray")

    h, w = g.shape
    if h < 3 or w < 3:
        return g.copy()

    divider = _find_divider_row(g)
    if divider <= 0 or divider >= h - 1:
        return np.zeros_like(g)

    patterns = _extract_upper_patterns(g, divider)
    if patterns is None:
        return np.zeros_like(g)

    full, damaged = _full_and_damaged(*patterns)
    diffs = _missing_offsets(full, damaged)
    if not diffs:
        return np.zeros_like(g)

    directions = _normalize_directions(diffs)
    if not directions:
        return np.zeros_like(g)

    sources = np.argwhere((g[divider + 1 :, :] == int(RED)).any(axis=0) == False)
    src_coords = np.argwhere(g[divider + 1 :, :] == int(RED))
    if src_coords.size == 0:
        return np.zeros_like(g)

    sr, sc = map(int, src_coords[0])
    out = np.zeros_like(g)
    out[divider + 1 :, :] = int(BLACK)

    # keep blue obstacles
    obs = _find_blue_obstacles(g)
    for r0, c0, r1, c1 in obs:
        out[r0 : r1 + 1, c0 : c1 + 1] = int(BLUE)

    out[sr, sc] = int(RED)
    obstacle_set = set((r, c) for rr0, cc0, rr1, cc1 in obs for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1))

    for dr, dc in directions:
        r, c = sr, sc
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            if r == divider:
                break
            if (r, c) in obstacle_set:
                break
            out[r, c] = int(RED)

    return out


def _find_divider_row(g: np.ndarray) -> int:
    counts = [int((g[r, :] == int(GRAY)).sum()) for r in range(g.shape[0])]
    if not counts:
        return -1
    return int(np.argmax(counts))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dr, dc in dirs:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _extract_upper_patterns(g: np.ndarray, divider: int) -> tuple[tuple[np.ndarray, tuple[int, int]], tuple[np.ndarray, tuple[int, int]]] | None:
    upper = g[:divider, :]
    mask = (upper != int(BLACK)) & (upper != int(GRAY))
    comps = _connected_components(mask)
    comps = [c for c in comps if len(c) >= 4]
    if len(comps) < 2:
        return None

    # pick two components with same aligned bbox size
    items = []
    for c in comps:
        rs = [p[0] for p in c]
        cs = [p[1] for p in c]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        crop = upper[r0 : r1 + 1, c0 : c1 + 1]
        bbox = (r0, c0, r1, c1)
        items.append((crop, bbox, int(mask[r0 : r1 + 1, c0 : c1 + 1].sum())))

    if len(items) < 2:
        return None

    # choose same-size pair if possible
    best: tuple = None
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, _ba, area_a = items[i]
            b, _bb, area_b = items[j]
            if a.shape == b.shape:
                if best is None or (area_a + area_b) > (best[2] + best[3]):
                    best = (items[i], items[j], area_a, area_b)
    if best is None:
        return None

    (a, ba, _), (b, bb, _), _, _ = best
    return a, b


def _full_and_damaged(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nonblack_a = int((a != int(BLACK)).sum())
    nonblack_b = int((b != int(BLACK)).sum())
    if nonblack_a >= nonblack_b:
        return a, b
    return b, a


def _missing_offsets(full: np.ndarray, damaged: np.ndarray) -> list[tuple[int, int]]:
    h, w = full.shape
    dh = max(h, damaged.shape[0])
    dw = max(w, damaged.shape[1])
    A = np.full((dh, dw), int(BLACK), dtype=int)
    B = np.full((dh, dw), int(BLACK), dtype=int)
    A[:h, :w] = full
    B[: damaged.shape[0], : damaged.shape[1]] = damaged

    out: list[tuple[int, int]] = []
    for r in range(dh):
        for c in range(dw):
            if A[r, c] != int(BLACK) and B[r, c] == int(BLACK):
                out.append((r, c))
    return out


def _normalize_directions(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not offsets:
        return []
    rs = [r for r, _ in offsets]
    cs = [c for _, c in offsets]
    center_r = int(round(sum(rs) / len(rs)))
    center_c = int(round(sum(cs) / len(cs)))
    dirs: set[tuple[int, int]] = set()

    for r, c in offsets:
        dr = r - center_r
        dc = c - center_c
        if dr == 0 and dc == 0:
            continue
        ad, ah = abs(dr), abs(dc)
        if ad > ah:
            d = (1 if dr > 0 else -1, 0)
        elif ah > ad:
            d = (0, 1 if dc > 0 else -1)
        else:
            if dr == 0 or dc == 0:
                continue
            d = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
        dirs.add(d)
    return list(dirs)


def _find_blue_obstacles(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = g.shape
    mask = g == int(BLUE)
    comps = _connected_components(mask)
    out: list[tuple[int, int, int, int]] = []
    for comp in comps:
        if len(comp) != 4:
            continue
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        if r1 - r0 == 1 and c1 - c0 == 1:
            out.append((r0, c0, r1, c1))
    return out


def generate_input() -> np.ndarray:
    h, w = 15, 15
    g = np.full((h, w), int(BLACK), dtype=int)

    divider = 7
    g[divider, :] = int(GRAY)

    # two 4x5-like patterns in upper area
    ph = random.randint(4, 5)
    pw = random.randint(4, 5)
    top = 2
    left1 = random.randint(1, 4)
    left2 = left1 + pw + 2
    if left2 + pw > w - 1:
        left2 = w - pw - 2
        left1 = left2 - pw - 2

    base = np.random.default_rng()
    palette = [Color.RED, Color.BLUE, Color.GREEN, Color.ORANGE, Color.MAROON]
    tile = np.array([[int(random.choice(palette)) for _ in range(pw)] for _ in range(ph)], dtype=int)
    for rr in range(top, top + ph):
        for cc in range(left1, left1 + pw):
            g[rr, cc] = int(tile[rr - top, cc - left1])

    full = g[top : top + ph, left1 : left1 + pw].copy()
    damage = full.copy()
    cells = [(rr, cc) for rr in range(ph) for cc in range(pw)]
    random.shuffle(cells)
    for rr, cc in cells[: random.randint(2, 4)]:
        damage[rr, cc] = int(BLACK)
    for rr in range(ph):
        for cc in range(pw):
            g[top : top + ph, left2 : left2 + pw][rr, cc] = damage[rr, cc]

    # place complete second pattern on right region by copying
    g[top : top + ph, left2 : left2 + pw] = np.array(damage, copy=False)
    # swap: make first complete and second damaged
    g[top : top + ph, left1 : left1 + pw] = full

    # red source in lower half
    sr = random.randint(divider + 1, h - 2)
    sc = random.randint(2, w - 3)
    g[sr, sc] = int(RED)

    # 2-4 blue 2x2 obstacles
    placed = 0
    tries = 0
    while placed < random.randint(2, 4) and tries < 200:
        tries += 1
        r0 = random.randint(divider + 1, h - 3)
        c0 = random.randint(0, w - 3)
        if g[r0 : r0 + 2, c0 : c0 + 2].any() and random.random() < 0.2:
            continue
        g[r0 : r0 + 2, c0 : c0 + 2] = int(BLUE)
        placed += 1

    # preserve at least one obstacle if random placed none
    if not np.any(g == int(BLUE)):
        g[divider + 2 : divider + 4, 2:4] = int(BLUE)

    return g
