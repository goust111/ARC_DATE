from __future__ import annotations

import numpy as np

BLACK = 0
RED = 2
BLUE = 1
ORANGE = 7
TEAL = 8


def _neighbors4(r: int, c: int, H: int, W: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < H:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < W:
        yield r, c + 1


def _components(mask: np.ndarray):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for rr, cc in _neighbors4(r, c, H, W):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    H, W = grid.shape
    out = np.zeros((16, 16), dtype=int)

    # upscale
    for r in range(H):
        for c in range(W):
            if grid[r, c] == RED:
                out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = RED
            else:
                out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = 0

    red_positions = np.argwhere(grid == RED)
    diag_values = set((int(r) - int(c)) for r, c in red_positions)
    for r in range(16):
        for c in range(16):
            or_r = r // 2
            or_c = c // 2
            if grid[or_r, or_c] == BLACK and (or_r - or_c) in diag_values:
                out[r, c] = BLUE if r % 2 == c % 2 else out[r, c]

    # preserve red blocks and black where not marked
    for r in range(H):
        for c in range(W):
            if grid[r, c] == BLACK and out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2][0, 0] != BLUE:
                pass

    black_mask = out == BLACK
    comps = _components(black_mask)
    if not comps:
        return out
    areas = [len(c) for c in comps]
    mn = min(areas)
    mx = max(areas)
    uniq_mn = areas.count(mn) == 1
    uniq_mx = areas.count(mx) == 1
    for comp in comps:
        if len(comp) == mn and uniq_mn:
            out[comp[:, 0], comp[:, 1]] = ORANGE
        elif len(comp) == mx and uniq_mx:
            out[comp[:, 0], comp[:, 1]] = TEAL
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    out = np.zeros((8, 8), dtype=int)
    positions = list((r, c) for r in range(8) for c in range(8) if r not in (0, 7) and c not in (0, 7))
    rng.shuffle(positions)
    for r, c in positions[:4]:
        out[r, c] = RED
    return out

