from collections import Counter, deque
import random

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _hole_count(inner):
    return len(_components(inner == 0))


def _band(a0, a1, b0, b1, width):
    lo, hi = max(a0, b0), min(a1, b1)
    mid = (lo + hi) // 2
    start = mid - width // 2
    return start, start + width - 1


def main(grid: np.ndarray) -> np.ndarray:
    """Order framed blocks by interior hole count, then connect neighbors with hole-count-colored bridges."""
    out = grid.copy()
    blocks = []
    for comp in _components(grid == 5):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        inner = grid[r0 + 1:r1, c0 + 1:c1]
        vals = [int(x) for x in inner.flat if x not in (0, 5)]
        if not vals:
            continue
        fill = Counter(vals).most_common(1)[0][0]
        blocks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "holes": _hole_count(inner),
                "width": min(r1 - r0 - 1, c1 - c0 - 1),
                "fill": fill,
            }
        )
    blocks.sort(key=lambda x: x["holes"])
    bridge_color = {1: 2, 2: 3, 3: 1, 4: 7}
    for a, b in zip(blocks, blocks[1:]):
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        color = bridge_color[a["holes"]]
        width = a["width"]
        if max(ar0, br0) <= min(ar1, br1):
            rr0, rr1 = _band(ar0 + 1, ar1 - 1, br0 + 1, br1 - 1, width)
            if ac1 < bc0:
                out[rr0:rr1 + 1, ac1 + 1:bc0] = color
            else:
                out[rr0:rr1 + 1, bc1 + 1:ac0] = color
        else:
            cc0, cc1 = _band(ac0 + 1, ac1 - 1, bc0 + 1, bc1 - 1, width)
            if ar1 < br0:
                out[ar1 + 1:br0, cc0:cc1 + 1] = color
            else:
                out[br1 + 1:ar0, cc0:cc1 + 1] = color
    out[grid != 0] = grid[grid != 0]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    placements = [(1, 1), (1, 12), (12, 12), (12, 1)]
    fills = random.sample([1, 2, 3, 7, 6], 4)
    hole_sets = [
        [(1, 1)],
        [(1, 0), (1, 2)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 0), (0, 2), (2, 0), (2, 2)],
    ]
    order = list(range(4))
    random.shuffle(order)
    for idx, (r0, c0) in enumerate(placements):
        holes = idx + 1
        fill = fills[idx]
        grid[r0:r0 + 5, c0:c0 + 5] = 5
        grid[r0 + 1:r0 + 4, c0 + 1:c0 + 4] = fill
        for hr, hc in hole_sets[holes - 1]:
            grid[r0 + 1 + hr, c0 + 1 + hc] = 0
    return grid
