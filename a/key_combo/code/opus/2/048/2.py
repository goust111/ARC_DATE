from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
TEAL = 8
COLORS = [1, 2, 3]


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


def _compress(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    h2 = H // 3
    w2 = W // 3
    out = np.zeros((h2, w2), dtype=int)
    for r in range(h2):
        for c in range(w2):
            block = grid[3 * r : 3 * r + 3, 3 * c : 3 * c + 3]
            colors = [int(v) for v in block.ravel() if int(v) not in (BLACK, GREY)]
            out[r, c] = colors[0] if colors else GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    small = _compress(grid)
    H, W = small.shape

    # Find 4-connected non-grey regions
    comps = _components(small != GREY)
    if not comps:
        return small

    areas = [len(c) for c in comps]
    min_area = min(areas)
    max_area = max(areas)
    min_ok = areas.count(min_area) == 1
    max_ok = areas.count(max_area) == 1
    out = small.copy()

    for comp in comps:
        if len(comp) == min_area and min_ok:
            out[comp[:, 0], comp[:, 1]] = ORANGE
        elif len(comp) == max_area and max_ok:
            out[comp[:, 0], comp[:, 1]] = TEAL
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    small = np.array(
        [
            [2, 2, 2, 2, 2, 2],
            [2, GREY, 2, 2, 2, 2],
            [2, 2, 2, GREY, 2, 2],
            [2, 2, GREY, 2, 2, 2],
            [2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, GREY],
        ],
        dtype=int,
    )

    # Inject some variation in color labels
    small = np.where(small == 2, 2 + (rng.integers(0, 2, size=small.shape)), small)
    small = np.where(small == 1, 1 + (rng.integers(0, 2, size=small.shape)), small)
    small[0, 5] = BLACK

    # Ensure valid colors only
    for r in range(6):
        for c in range(6):
            if small[r, c] not in (GREY, 1, 2, 3, BLACK):
                small[r, c] = int(rng.choice(COLORS))
    small = small % 4
    small[0:1] = GREY
    small[:, 0] = GREY
    small[5, :] = GREY

    # upscale with small amount of grey noise
    out = np.zeros((18, 18), dtype=int)
    for r in range(6):
        for c in range(6):
            color = int(small[r, c])
            block = 3 * r
            blk = c * 3
            if color == GREY:
                arr = np.full((3, 3), GREY, dtype=int)
                for _ in range(2):
                    rr = int(rng.integers(0, 3))
                    cc = int(rng.integers(0, 3))
                    arr[rr, cc] = BLACK
                out[block : block + 3, blk : blk + 3] = arr
            else:
                arr = np.full((3, 3), color, dtype=int)
                noise = rng.integers(0, 3)
                for _ in range(int(noise)):
                    rr = int(rng.integers(0, 3))
                    cc = int(rng.integers(0, 3))
                    arr[rr, cc] = GREY
                out[block : block + 3, blk : blk + 3] = arr
    return out

