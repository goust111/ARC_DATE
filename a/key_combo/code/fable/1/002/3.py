import random
from collections import deque

import numpy as np


BG = 0
BODY = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
        comps.append(comp)
    return comps


def _sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def _sprite_infos(g: np.ndarray):
    colors = sorted(set(map(int, g.ravel())) - {BG})
    body = max(colors, key=lambda c: int(np.sum(g == c)))
    marker_cells = [(int(r), int(c), int(g[r, c])) for r, c in zip(*np.where((g != BG) & (g != body)))]
    sprite_mask = g != BG
    infos = []
    used_markers = set()
    for mr, mc, color in marker_cells:
        mask = (g == body).copy()
        mask[mr, mc] = True
        comp = next(comp for comp in _components(mask) if (mr, mc) in comp)
        body_cells = [(r, c) for r, c in comp if g[r, c] == body]
        cr = sum(r for r, _ in body_cells) / len(body_cells)
        cc = sum(c for _, c in body_cells) / len(body_cells)
        dr, dc = -_sign(cr - mr), -_sign(cc - mc)
        assert dr != 0 and dc != 0
        infos.append((mr, mc, color, dr, dc))
        used_markers.add((mr, mc))
    return body, sprite_mask, infos


def _trace(out: np.ndarray, sprite_mask: np.ndarray, r: int, c: int, dr: int, dc: int, color: int):
    seen = set()
    for _ in range(out.size * 8):
        nr, nc = r + dr, c + dc
        bounced = False
        if nr < 0 or nr >= out.shape[0]:
            dr = -dr
            nr = r + dr
            bounced = True
        if nc < 0 or nc >= out.shape[1]:
            dc = -dc
            nc = c + dc
            bounced = True
        state = (r, c, dr, dc)
        if state in seen and not bounced:
            break
        seen.add(state)
        r, c = nr, nc
        if sprite_mask[r, c]:
            break
        if out[r, c] == BG:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire one diagonal ray from each marked sprite. The ray direction is opposite
    the body-centroid bias from the marker, and grid borders reflect the ray
    until it enters any sprite pixel.
    """
    g = np.asarray(grid, dtype=int)
    _, sprite_mask, infos = _sprite_infos(g)
    out = g.copy()
    for mr, mc, color, dr, dc in sorted(infos):
        _trace(out, sprite_mask, mr, mc, dr, dc, color)
    return out


def _shape_for_dir(dr, dc):
    return [(0, 0), (-dr, 0), (0, -dc), (-dr, -dc), (-2 * dr, -dc), (-dr, -2 * dc)]


def _place_sprite(grid, center, direction, marker_color):
    dr, dc = direction
    rel = _shape_for_dir(dr, dc)
    cells = [(center[0] + r, center[1] + c) for r, c in rel]
    all_cells = cells + [center]
    for r, c in all_cells:
        if not (1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1) or grid[r, c] != BG:
            return False
    for r, c in cells:
        grid[r, c] = BODY
    grid[center] = marker_color
    return True


def _ray_cells(grid, sprite_mask, start, direction):
    r, c = start
    dr, dc = direction
    cells = []
    seen = set()
    for _ in range(grid.size * 8):
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= grid.shape[0]:
            dr = -dr
            nr = r + dr
        if nc < 0 or nc >= grid.shape[1]:
            dc = -dc
            nc = c + dc
        state = (r, c, dr, dc)
        if state in seen:
            return None
        seen.add(state)
        r, c = nr, nc
        if sprite_mask[r, c]:
            return cells
        cells.append((r, c))
    return None


def _gen_attempt():
    h = w = random.randint(17, 25)
    grid = np.zeros((h, w), dtype=int)
    n = random.randint(3, 5)
    dirs = random.choices([(-1, -1), (-1, 1), (1, -1), (1, 1)], k=n)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    centers = []
    for direction, color in zip(dirs, colors):
        for _ in range(200):
            center = (random.randint(4, h - 5), random.randint(4, w - 5))
            if all(abs(center[0] - r) + abs(center[1] - c) > 6 for r, c in centers) and _place_sprite(grid, center, direction, color):
                centers.append(center)
                break
        else:
            return None
    try:
        _, sprite_mask, infos = _sprite_infos(grid)
        paths = []
        occupied = set()
        for mr, mc, color, dr, dc in infos:
            cells = _ray_cells(grid, sprite_mask, (mr, mc), (dr, dc))
            if cells is None or not cells or occupied & set(cells):
                return None
            occupied.update(cells)
            paths.append(cells)
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
