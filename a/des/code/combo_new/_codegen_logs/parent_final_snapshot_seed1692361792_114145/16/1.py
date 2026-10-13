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
            border = {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)}
            border |= {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
            if not set(cells).issubset(border):
                continue
            miss = {
                "up": [c for c in range(c0 + 1, c1) if grid[r0, c] == BLACK],
                "right": [r for r in range(r0 + 1, r1) if grid[r, c1] == BLACK],
                "down": [c for c in range(c0 + 1, c1) if grid[r1, c] == BLACK],
                "left": [r for r in range(r0 + 1, r1) if grid[r, c0] == BLACK],
            }
            gap = [k for k, v in miss.items() if len(v) >= 2]
            if len(gap) == 1:
                frames.append((color, gap[0], r0, r1, c0, c1))
    return frames


def build_structure(sprite, frame_color):
    colors = [v for v in np.unique(sprite) if v != BLACK]
    sprite_color = colors[0]
    h, w = sprite.shape
    base = max(h, w)
    n = int(np.count_nonzero(sprite))
    size = base + 2 * n
    out = np.zeros((size, size), dtype=int)
    br = n + (base - h) // 2
    bc = n + (base - w) // 2
    out[br : br + h, bc : bc + w][sprite != BLACK] = sprite_color
    for layer in range(1, n + 1):
        off = n - layer
        side = base + 2 * layer
        color = frame_color if layer % 2 == 1 else sprite_color
        out[off, off : off + side] = color
        out[off + side - 1, off : off + side] = color
        out[off : off + side, off] = color
        out[off : off + side, off + side - 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    pieces = []
    rot_map = {"up": 0, "right": -1, "down": 2, "left": 1}
    for frame_color, gap, r0, r1, c0, c1 in find_frames(grid):
        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        mask = (inside != BLACK) & (inside != frame_color)
        cells = np.argwhere(mask)
        if len(cells) == 0:
            continue
        sr0, sc0 = cells.min(axis=0)
        sr1, sc1 = cells.max(axis=0)
        sprite = inside[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        sprite = np.rot90(sprite, rot_map[gap])
        pieces.append((r0, c0, build_structure(sprite, frame_color)))
    pieces.sort(key=lambda t: (t[0], t[1]))
    if not pieces:
        return np.zeros((1, 1), dtype=int)
    height = max(p.shape[0] for _, _, p in pieces)
    width = sum(p.shape[1] for _, _, p in pieces) + len(pieces) - 1
    out = np.zeros((height, width), dtype=int)
    x = 0
    for _, _, piece in pieces:
        out[: piece.shape[0], x : x + piece.shape[1]] = piece
        x += piece.shape[1] + 1
    return out


def rand_sprite():
    h, w = random.randint(2, 3), random.randint(2, 4)
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


def generate_input() -> np.ndarray:
    h, w = random.randint(12, 16), random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    x = 1
    for _ in range(random.randint(1, 2)):
        side = random.randint(5, 6)
        r0 = random.randint(1, h - side - 1)
        c0 = x
        if c0 + side > w - 2:
            break
        frame_color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
        sprite_color = random.choice([v for v in [1, 2, 3, 4, 6, 7, 8, 9] if v != frame_color])
        r1, c1 = r0 + side - 1, c0 + side - 1
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        gap = random.choice(["up", "right", "down", "left"])
        gap_len = 2
        if gap in ("up", "down"):
            start = c0 + 1 + random.randint(0, side - gap_len - 2)
            grid[r0 if gap == "up" else r1, start : start + gap_len] = BLACK
        else:
            start = r0 + 1 + random.randint(0, side - gap_len - 2)
            grid[start : start + gap_len, c0 if gap == "left" else c1] = BLACK
        sprite = rand_sprite() * sprite_color
        while np.count_nonzero(sprite) > 4:
            sprite = rand_sprite() * sprite_color
        sh, sw = sprite.shape
        grid[r0 + 1 : r0 + 1 + sh, c0 + 1 : c0 + 1 + sw][sprite != 0] = sprite_color
        x += side + 3
    return grid
