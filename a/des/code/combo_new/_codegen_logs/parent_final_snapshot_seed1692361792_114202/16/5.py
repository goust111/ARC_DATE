import random
from collections import deque
import numpy as np

BLACK = 0


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


def find_frames(grid):
    frames = []
    for color in [v for v in np.unique(grid) if v != BLACK]:
        for cells in components(grid == color):
            r0, r1, c0, c1 = bbox(cells)
            if r1 - r0 != c1 - c0 or r1 - r0 < 4:
                continue
            side = r1 - r0 + 1
            border = {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)}
            border |= {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
            if not set(cells).issubset(border):
                continue
            miss = {
                "up": [c for c in range(c0 + 1, c1) if grid[r0, c] == BLACK],
                "down": [c for c in range(c0 + 1, c1) if grid[r1, c] == BLACK],
                "left": [r for r in range(r0 + 1, r1) if grid[r, c0] == BLACK],
                "right": [r for r in range(r0 + 1, r1) if grid[r, c1] == BLACK],
            }
            gap = [k for k, v in miss.items() if len(v) >= 2]
            if len(gap) == 1:
                frames.append((color, gap[0], r0, r1, c0, c1))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    for frame_color, gap, r0, r1, c0, c1 in find_frames(grid):
        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        mask = (inside != BLACK) & (inside != frame_color)
        cells = np.argwhere(mask)
        if len(cells) == 0:
            continue
        sr0, sc0 = cells.min(axis=0)
        sr1, sc1 = cells.max(axis=0)
        sprite = inside[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        sprite = np.where(sprite != BLACK, frame_color, BLACK)
        if gap in ("left", "right"):
            sprite = np.fliplr(sprite)
        else:
            sprite = np.flipud(sprite)
        out[r0 + 1 : r1, c0 + 1 : c1][mask] = BLACK
        sh, sw = sprite.shape
        if gap == "up":
            rr = r1 + 1
            cc = (c0 + c1 - sw + 1) // 2
        elif gap == "down":
            rr = r0 - sh
            cc = (c0 + c1 - sw + 1) // 2
        elif gap == "left":
            rr = (r0 + r1 - sh + 1) // 2
            cc = c1 + 1
        else:
            rr = (r0 + r1 - sh + 1) // 2
            cc = c0 - sw
        for r in range(sh):
            for c in range(sw):
                if sprite[r, c] != BLACK:
                    out[rr + r, cc + c] = frame_color
    return out


def rand_sprite(size):
    h = random.randint(2, size)
    w = random.randint(2, size)
    mask = np.zeros((h, w), dtype=int)
    cells = [(0, 0)]
    mask[0, 0] = 1
    target = random.randint(2, min(6, h * w))
    while np.count_nonzero(mask) < target:
        r, c = random.choice(cells)
        nr, nc = r + random.choice([-1, 1, 0, 0]), c + random.choice([0, 0, -1, 1])
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] == 0:
            mask[nr, nc] = 1
            cells.append((nr, nc))
    return mask


def _generate_once() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)
    x = 1
    for _ in range(random.randint(2, 3)):
        side = random.randint(6, 8)
        gap = random.choice(["up", "down", "left", "right"])
        frame_color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
        sprite_color = random.choice([c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c != frame_color])
        r0 = random.randint(5, h - side - 5)
        c0 = x
        c1 = c0 + side - 1
        r1 = r0 + side - 1
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        gap_len = random.choice([2, 3])
        if gap in ("up", "down"):
            start = c0 + (side - gap_len) // 2
            grid[r0 if gap == "up" else r1, start : start + gap_len] = BLACK
        else:
            start = r0 + (side - gap_len) // 2
            grid[start : start + gap_len, c0 if gap == "left" else c1] = BLACK
        sprite = rand_sprite(4) * sprite_color
        sh, sw = sprite.shape
        ir = r0 + 1 + (side - 2 - sh) // 2
        ic = c0 + 1 + (side - 2 - sw) // 2
        grid[ir : ir + sh, ic : ic + sw][sprite != 0] = sprite_color
        x += side + 3
        if x + side >= w - 1:
            break
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_once()
        try:
            out = main(grid)
        except Exception:
            continue
        if out.ndim == 2 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate valid input")
