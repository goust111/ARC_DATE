import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def _bbox(pts):
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """Transfer right-side shapes into a left frame by reusing marker-to-shape offsets.

    Input structure: a maroon vertical divider separates a left grey target
    frame with colored anchors from right-side colored shapes, each paired with
    an isolated same-colored marker pixel.
    Rule: for each color, compute the source shape center relative to its marker,
    apply the same offset to the matching frame anchor, stamp the shape inside
    the frame with clipping, and return only the left side without the divider.
    Key insight: the marker is a coordinate reference, and its offset from the
    shape is transferred to the same-colored anchor.
    """
    divs = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 9)]
    div = divs[0] if divs else grid.shape[1] // 2
    out = grid[:, :div].copy()
    grey = 5
    frame = max(_components(out == grey), key=len)
    fr1, fr2, fc1, fc2 = _bbox(frame)
    anchors = {}
    for r in range(fr1 + 1, fr2):
        for c in range(fc1 + 1, fc2):
            if out[r, c] not in (0, grey):
                anchors[int(out[r, c])] = (r, c)
    right = grid[:, div + 1:]
    for color in anchors:
        comps = _components(right == color)
        if len(comps) < 2:
            continue
        marker = min(comps, key=len)
        shape = max(comps, key=len)
        mr, mc = marker[0]
        sr1, sr2, sc1, sc2 = _bbox(shape)
        center = ((sr1 + sr2) // 2, (sc1 + sc2) // 2)
        off = (center[0] - mr, center[1] - mc)
        target_center = (anchors[color][0] + off[0], anchors[color][1] + off[1])
        for r, c in shape:
            nr = target_center[0] + (r - center[0])
            nc = target_center[1] + (c - center[1])
            if fr1 < nr < fr2 and fc1 < nc < fc2:
                out[nr, nc] = color
    return out


def generate_input() -> np.ndarray:
    h, left_w, right_w = 18, 12, 13
    grid = np.zeros((h, left_w + 1 + right_w), dtype=int)
    div = left_w
    grid[:, div] = 9
    r1, r2, c1, c2 = 2, 15, 2, 10
    grid[r1:r2 + 1, c1] = 5
    grid[r1:r2 + 1, c2] = 5
    grid[r1, c1:c2 + 1] = 5
    grid[r2, c1:c2 + 1] = 5
    colors = random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(2, 4))
    anchor_spots = [(r1 + 1, c1 + 2), (r1 + 2, c2 - 1), (r2 - 1, c1 + 3), (r2 - 2, c2 - 2)]
    random.shuffle(anchor_spots)
    shapes = [[(0, 1), (1, 0), (1, 1), (2, 1)], [(0, 0), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (1, 0), (1, 1), (1, 2)]]
    for i, color in enumerate(colors):
        ar, ac = anchor_spots[i]
        grid[ar, ac] = color
        shape = shapes[i % len(shapes)]
        sr = 2 + (i % 2) * 7
        sc = div + 2 + (i // 2) * 6
        for a, b in shape:
            grid[sr + a, sc + b] = color
        cr = sr + 1
        cc = sc + 1
        mr = cr + random.choice([-2, 2])
        mc = cc + random.choice([-2, 2])
        grid[mr, mc] = color
    return grid
