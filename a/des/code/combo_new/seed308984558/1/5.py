import random
from collections import deque

import numpy as np


BG = 0


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each stamp into the frame whose border color matches the stamp's marker pixel.

    Input structure:
    - A black background contains small monochrome stamp sprites and hollow rectangular target frames.
    - Each stamp has a one-pixel marker directly above its top-left corner.
    - Every frame interior size matches one stamp bounding box, and frame colors are unique.

    Rule:
    - Read the marker color above each stamp to choose the target frame with the same border color.
    - Copy the stamp shape into that frame's interior, aligned to the interior top-left corner.
    - Recolor the copied pixels to the frame color and keep the original stamps, markers, and frames unchanged.

    Key insight:
    - The marker color determines the destination and final copy color, while only the stamp shape is transferred.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    frames = {}
    stamps = []
    colors = [v for v in np.unique(g) if v != BG]
    for color in colors:
        for cells in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(cells)
            if len(cells) == 2 * (r1 - r0 + 1) + 2 * (c1 - c0 - 1):
                frames[int(color)] = (r0, c0, r1, c1)
            else:
                if r0 > 0 and g[r0 - 1, c0] != BG:
                    stamps.append((int(color), cells, int(g[r0 - 1, c0])))
    for _stamp_color, cells, target_color in stamps:
        r0, c0, r1, c1 = _bbox(cells)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        fr0, fc0, fr1, fc1 = frames[target_color]
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask[r, c]:
                    out[fr0 + 1 + r, fc0 + 1 + c] = target_color
    return out


SHAPES = [
    np.array([[1, 1], [1, 0]]),
    np.array([[1, 0, 1], [1, 1, 1]]),
    np.array([[1, 1, 1], [0, 1, 0]]),
]


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 28), random.randint(18, 28)
    g = np.zeros((h, w), dtype=int)
    pairs = random.randint(2, 4)
    frame_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], pairs)
    stamp_colors = random.sample([v for v in [1, 2, 3, 4, 6, 7, 8, 9] if v not in frame_colors], pairs)
    for fcol, scol in zip(frame_colors, stamp_colors):
        shape = random.choice(SHAPES)
        sh, sw = shape.shape
        placed_stamp = False
        for _ in range(200):
            r0 = random.randint(1, h - sh - 1)
            c0 = random.randint(0, w - sw)
            if np.any(g[max(0, r0 - 2) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)]):
                continue
            g[r0 - 1, c0] = fcol
            for r in range(sh):
                for c in range(sw):
                    if shape[r, c]:
                        g[r0 + r, c0 + c] = scol
            placed_stamp = True
            break
        if not placed_stamp:
            return generate_input()
        fh, fw = sh + 2, sw + 2
        placed_frame = False
        for _ in range(200):
            r0 = random.randint(0, h - fh)
            c0 = random.randint(0, w - fw)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + fh + 1), max(0, c0 - 1) : min(w, c0 + fw + 1)]):
                continue
            g[r0 : r0 + fh, c0] = g[r0 : r0 + fh, c0 + fw - 1] = fcol
            g[r0, c0 : c0 + fw] = g[r0 + fh - 1, c0 : c0 + fw] = fcol
            placed_frame = True
            break
        if not placed_frame:
            return generate_input()
    return g
