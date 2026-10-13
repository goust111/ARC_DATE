import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    patterns = []
    for comp in _components(grid != 0):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r1 - r0 == 4 and c1 - c0 == 4:
            inner = grid[r0 + 1:r0 + 4, c0 + 1:c0 + 4]
            vals, cnts = np.unique(inner, return_counts=True)
            inner_color = int(vals[np.argmax(cnts)])
            frame_vals = [grid[r, c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                          if r in (r0, r1) or c in (c0, c1)]
            frame_color = max((v for v in frame_vals if v != 9), key=frame_vals.count)
            patterns.append((r0, c0, inner_color, int(frame_color)))
    patterns.sort()
    for r0, c0, inner_color, frame_color in patterns:
        marks = []
        for r in range(r0, r0 + 5):
            for c in range(c0, c0 + 5):
                edge = r in (r0, r0 + 4) or c in (c0, c0 + 4)
                if not edge:
                    out[r, c] = frame_color
                elif grid[r, c] == 9:
                    marks.append((r, c))
                    out[r, c] = 9
                else:
                    out[r, c] = inner_color
        for r, c in marks:
            dr = -1 if r == r0 else 1
            dc = -1 if c == c0 else 1
            x, y = r + dr, c + dc
            while 0 <= x < out.shape[0] and 0 <= y < out.shape[1]:
                if grid[x, y] == 0:
                    out[x, y] = frame_color
                x += dr
                y += dc
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(13, 17)
    g = np.zeros((h, w), dtype=int)
    locs = [(random.randint(3, 5), random.randint(3, 5))]
    if random.random() < 0.45:
        locs.append((random.randint(7, h - 6), random.randint(7, w - 6)))
    colors = [(1, 3), (4, 6), (2, 8)]
    corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
    for i, (r0, c0) in enumerate(locs):
        inner, frame = colors[i % len(colors)]
        g[r0:r0 + 5, c0:c0 + 5] = frame
        g[r0 + 1:r0 + 4, c0 + 1:c0 + 4] = inner
        for cr, cc in random.sample(corners, random.randint(1, 2)):
            g[r0 + cr, c0 + cc] = 9
    return g
