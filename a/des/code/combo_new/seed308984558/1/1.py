import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5


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


def _shape_key(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        mask[r - r0, c - c0] = 1
    keys = []
    for k in range(4):
        rot = np.rot90(mask, k)
        keys.append(rot.tobytes())
    return min(keys)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow frame from the matching outside shape and mirror the inside trap.

    Input structure:
    - A black background contains several one-pixel-thick grey rectangular frames.
    - Each frame encloses one colored trap sprite, separated from the border.
    - Outside the frames, one candidate sprite per frame appears in a unique color.

    Rule:
    - Match every trap sprite to the outside candidate whose shape matches up to rotation.
    - Repaint that frame border to the candidate color and remove the outside candidate.
    - Mirror the trap sprite left-to-right inside its own bounding box.

    Key insight:
    - The inside sprite chooses the border color by rotation-invariant shape matching, and that same sprite is also mirrored.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    frame_comps = []
    for cells in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(cells)
        if len(cells) == 2 * (r1 - r0 + 1) + 2 * (c1 - c0 - 1):
            frame_comps.append((cells, (r0, c0, r1, c1)))
    inside_mask = np.zeros_like(g, dtype=bool)
    traps = []
    for frame_cells, (r0, c0, r1, c1) in frame_comps:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = [v for v in np.unique(inner) if v != BG]
        assert len(colors) == 1
        color = int(colors[0])
        cells = _components(inner == color)[0]
        cells = [(r0 + 1 + r, c0 + 1 + c) for r, c in cells]
        traps.append((cells, frame_cells))
        inside_mask[r0 + 1 : r1, c0 + 1 : c1] = True
    candidates = []
    for color in [v for v in np.unique(g) if v not in (BG, FRAME)]:
        for cells in _components(g == int(color)):
            if any(inside_mask[r, c] for r, c in cells):
                continue
            candidates.append((int(color), cells))
    by_key = {_shape_key(cells): (color, cells) for color, cells in candidates}
    for trap_cells, frame_cells in traps:
        color, cand_cells = by_key[_shape_key(trap_cells)]
        for r, c in cand_cells:
            out[r, c] = BG
        for r, c in frame_cells:
            out[r, c] = color
        r0, c0, r1, c1 = _bbox(trap_cells)
        box = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        trap_color = out[trap_cells[0][0], trap_cells[0][1]]
        for r, c in trap_cells:
            box[r - r0, c - c0] = trap_color
            out[r, c] = BG
        box = np.fliplr(box)
        for r in range(box.shape[0]):
            for c in range(box.shape[1]):
                if box[r, c]:
                    out[r0 + r, c0 + c] = trap_color
    return out


SHAPES = [
    np.array([[1, 1], [1, 0]]),
    np.array([[1, 1, 1], [0, 1, 0]]),
    np.array([[1, 1, 1]]),
    np.array([[1, 0], [1, 1], [1, 0]]),
]


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 28), random.randint(18, 28)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    shapes = random.sample(SHAPES, len(colors))
    frames = []
    forbidden = np.zeros((h, w), dtype=bool)
    for shape, color in zip(shapes, colors):
        ih = max(shape.shape[0] + 2, random.randint(shape.shape[0] + 2, shape.shape[0] + 4))
        iw = max(shape.shape[1] + 2, random.randint(shape.shape[1] + 2, shape.shape[1] + 4))
        fh, fw = ih + 2, iw + 2
        for _ in range(200):
            r0 = random.randint(0, h - fh)
            c0 = random.randint(0, w - fw)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + fh + 1), max(0, c0 - 1) : min(w, c0 + fw + 1)]):
                continue
            g[r0 : r0 + fh, c0] = g[r0 : r0 + fh, c0 + fw - 1] = FRAME
            g[r0, c0 : c0 + fw] = g[r0 + fh - 1, c0 : c0 + fw] = FRAME
            tr = r0 + 1 + random.randint(0, ih - shape.shape[0])
            tc = c0 + 1 + random.randint(0, iw - shape.shape[1])
            trap_color = random.choice([v for v in [1, 2, 3, 4, 6, 7, 8, 9] if v not in colors + [FRAME]])
            for r in range(shape.shape[0]):
                for c in range(shape.shape[1]):
                    if shape[r, c]:
                        g[tr + r, tc + c] = trap_color
            frames.append((shape, color))
            forbidden[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = True
            break
    for shape, color in frames:
        sprite = np.rot90(shape, random.randint(0, 3))
        sh, sw = sprite.shape
        for _ in range(200):
            r0 = random.randint(0, h - sh)
            c0 = random.randint(0, w - sw)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)]):
                continue
            if np.any(forbidden[r0 : r0 + sh, c0 : c0 + sw]):
                continue
            for r in range(sh):
                for c in range(sw):
                    if sprite[r, c]:
                        g[r0 + r, c0 + c] = color
            break
    return g
