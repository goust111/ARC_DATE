import random
from collections import deque

import numpy as np


def _components(grid: np.ndarray, blocked: set[int]) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) in blocked:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) not in blocked:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Route red->yellow->green horizontally-first, then stamp the source sprite at each bend."""
    out = grid.copy()
    sprite = max((comp for comp in _components(grid, {0, 2, 3, 4}) if np.any(grid[tuple(zip(*comp))] == 5)), key=len)
    anchor = next((r, c) for r, c in sprite if int(grid[r, c]) == 5)
    rel_pixels = [(r - anchor[0], c - anchor[1], int(grid[r, c])) for r, c in sprite if int(grid[r, c]) != 5]

    markers = {}
    for color in (2, 4, 3):
        pos = np.argwhere(grid == color)
        if pos.size != 2:
            raise ValueError("expected a single waypoint per color")
        markers[color] = tuple(map(int, pos[0]))

    bends = []
    for src_color, dst_color in ((2, 4), (4, 3)):
        sr, sc = markers[src_color]
        tr, tc = markers[dst_color]
        bends.append((sr, tc))
        step = 1 if tc > sc else -1
        for cc in range(sc + step, tc + step, step):
            if out[sr, cc] == 0:
                out[sr, cc] = 5
        step = 1 if tr > sr else -1
        for rr in range(sr + step, tr + step, step):
            if out[rr, tc] == 0:
                out[rr, tc] = 5

    for br, bc in bends:
        for dr, dc, color in rel_pixels:
            rr, cc = br + dr, bc + dc
            if out[rr, cc] not in (2, 3, 4):
                out[rr, cc] = color
    return out


def _random_sprite(rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int], dict[tuple[int, int], int]]:
    size = rng.randint(4, 6)
    cells = [(0, 0)]
    seen = {(0, 0)}
    while len(cells) < size:
        r, c = rng.choice(cells)
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        rng.shuffle(nbrs)
        for nr, nc in nbrs:
            if (nr, nc) not in seen and -1 <= nr <= 1 and -1 <= nc <= 1:
                seen.add((nr, nc))
                cells.append((nr, nc))
                break
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = [(r - min_r, c - min_c) for r, c in cells]
    anchor = rng.choice(norm)
    palette = rng.sample([1, 6, 7, 8, 9], k=rng.randint(2, 3))
    colors = {}
    for cell in norm:
        colors[cell] = 5 if cell == anchor else rng.choice(palette)
    return norm, anchor, colors


def generate_input() -> np.ndarray:
    rng = random.Random()
    while True:
        h = w = rng.choice([13, 15, 17])
        grid = np.zeros((h, w), dtype=int)
        cells, anchor, colors = _random_sprite(rng)
        max_sr = h - max(r for r, _ in cells) - 1
        max_sc = w - max(c for _, c in cells) - 1
        sr = rng.randint(1, max_sr - 5)
        sc = rng.randint(1, max_sc - 5)
        sprite_abs = {(sr + r, sc + c): color for (r, c), color in colors.items()}
        for (r, c), color in sprite_abs.items():
            grid[r, c] = color

        candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if grid[r, c] == 0]
        rng.shuffle(candidates)
        chosen = None
        for red in candidates:
            for yellow in candidates:
                if yellow == red or red[0] == yellow[0] or red[1] == yellow[1]:
                    continue
                for green in candidates:
                    if green in (red, yellow) or yellow[0] == green[0] or yellow[1] == green[1]:
                        continue
                    bends = [(red[0], yellow[1]), (yellow[0], green[1])]
                    if len(set(bends)) != 2:
                        continue
                    ok = True
                    occupied = set(sprite_abs)
                    for br, bc in bends:
                        for r, c in colors:
                            rr = br + r - anchor[0]
                            cc = bc + c - anchor[1]
                            if not (0 <= rr < h and 0 <= cc < w):
                                ok = False
                                break
                            if (rr, cc) in {red, yellow, green}:
                                continue
                            if (rr, cc) in occupied:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        chosen = (red, yellow, green)
                        break
                if chosen:
                    break
            if chosen:
                break
        if not chosen:
            continue

        for color, pos in zip((2, 4, 3), chosen):
            grid[pos] = color
        return grid
