import random
from collections import deque
import numpy as np

BLACK = 0
GREY = 5


def components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
                comps.append(cells)
    return comps


def bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def side_of_key(arr, key):
    pos = np.argwhere(arr == key)[0]
    if pos[0] == 0:
        return "up"
    if pos[0] == arr.shape[0] - 1:
        return "down"
    if pos[1] == 0:
        return "left"
    return "right"


def opposite(side):
    return {"up": "down", "down": "up", "left": "right", "right": "left"}[side]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    grey = np.argwhere(grid == GREY)
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    inner = grid[r0 : r1 + 1, c0 : c1 + 1]
    black_comps = components(inner == BLACK)
    slots = []
    lock_positions = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r, c] not in (BLACK, GREY):
                lock_positions.append((r, c, grid[r, c]))
    for lr, lc, color in lock_positions:
        for comp in black_comps:
            cells = {(r0 + r, c0 + c) for r, c in comp}
            if any(abs(lr - r) + abs(lc - c) == 1 for r, c in cells):
                sr0, sr1, sc0, sc1 = bbox(list(cells))
                if sr0 == r0:
                    side = "up"
                elif sr1 == r1:
                    side = "down"
                elif sc0 == c0:
                    side = "left"
                else:
                    side = "right"
                slots.append((color, side, sr0, sr1, sc0, sc1, lr, lc))
                break
    piece_comps = []
    for cells in components((grid != BLACK) & (grid != GREY)):
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells):
            continue
        pr0, pr1, pc0, pc1 = bbox(cells)
        crop = grid[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        vals = [grid[r, c] for r, c in cells]
        key = min(set(vals), key=lambda v: vals.count(v))
        piece_comps.append((key, crop, (pr0, pr1, pc0, pc1)))
    out = grid.copy()
    for _, _, (pr0, pr1, pc0, pc1) in piece_comps:
        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = BLACK
    for key, side, sr0, sr1, sc0, sc1, lr, lc in slots:
        matches = [p for p in piece_comps if p[0] == key]
        if not matches:
            continue
        piece = matches[0]
        _, crop, _ = piece
        for k in range(4):
            rot = np.rot90(crop, k)
            if rot.shape != (sr1 - sr0 + 1, sc1 - sc0 + 1):
                continue
            if side_of_key(rot, key) != opposite(side):
                continue
            placed = rot.copy()
            placed[placed == key] = BLACK
            out[sr0 : sr1 + 1, sc0 : sc1 + 1] = placed
            out[lr, lc] = BLACK
            break
    return out


def rotate_side(side, k):
    order = ["up", "left", "down", "right"]
    return order[(order.index(side) + k) % 4]


def tile_with_key(base, key, h, w, side):
    arr = np.full((h, w), base, dtype=int)
    if h >= 3:
        arr[1:-1, w // 2] = random.choice([BLACK, base])
        arr[h // 2, 1:-1] = base
    if side == "up":
        arr[0, w // 2] = key
    elif side == "down":
        arr[-1, w // 2] = key
    elif side == "left":
        arr[h // 2, 0] = key
    else:
        arr[h // 2, -1] = key
    return arr


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(22, 28)
    grid = np.zeros((h, w), dtype=int)
    bh, bw = random.randint(8, 10), random.randint(10, 12)
    r0 = (h - bh) // 2
    c0 = (w - bw) // 2
    grid[r0 : r0 + bh, c0 : c0 + bw] = GREY
    keys = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    piece_spots = [(1, 1), (1, w - 6), (h - 6, 1), (h - 6, w - 6)]
    random.shuffle(piece_spots)
    idx = 0
    for key in keys:
        side = random.choice(["up", "down", "left", "right"])
        sh, sw = random.randint(3, 4), random.randint(3, 4)
        if side in ("up", "down"):
            x = random.randint(c0 + 1, c0 + bw - sw - 1)
            y = r0 if side == "up" else r0 + bh - sh
            grid[y : y + sh, x : x + sw] = BLACK
            grid[y if side == "up" else y + sh - 1, x + sw // 2] = key
        else:
            y = random.randint(r0 + 1, r0 + bh - sh - 1)
            x = c0 if side == "left" else c0 + bw - sw
            grid[y : y + sh, x : x + sw] = BLACK
            grid[y + sh // 2, x if side == "left" else x + sw - 1] = key
        base = random.choice([v for v in [1, 2, 3, 4, 6, 7, 8, 9] if v != key])
        desired = opposite(side)
        tile = tile_with_key(base, key, sh, sw, desired)
        k = random.randint(0, 3)
        tile = np.rot90(tile, k)
        th, tw = tile.shape
        pr, pc = piece_spots[idx]
        pr = min(pr, h - th - 1)
        pc = min(pc, w - tw - 1)
        idx += 1
        grid[pr : pr + th, pc : pc + tw] = np.where(tile != BLACK, tile, grid[pr : pr + th, pc : pc + tw])
    return grid
