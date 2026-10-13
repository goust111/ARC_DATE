import random
from collections import deque

import numpy as np

BLACK = 0
RED = 2


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _receiver_info(grid: np.ndarray):
    parts = []
    for cells in _components((grid != BLACK) & (grid != RED)):
        color = int(grid[cells[0]])
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        parts.append((min(rows), max(rows), min(cols), max(cols), color))
    return parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast red rays from the tallest tower and let contacted receiver squares emit vertical rays.

    Input structure:
    - Black background with several red vertical bars and several solid non-red square receivers.
    - Exactly one red bar is tallest, breaking ties by leftmost position.
    - Receivers are isolated from towers and from one another.

    Rule:
    - Choose the tallest red bar as the active tower and extend red horizontal rays left and right from every tower pixel until the next non-black cell or the grid edge.
    - If a ray stops immediately beside a receiver square, activate that receiver.
    - Each activated receiver emits same-colored vertical rays of length equal to its side length from every top-edge pixel upward and every bottom-edge pixel downward, without overwriting non-black pixels.

    Key insight:
    - Only one tower is active, and receiver output length comes from the receiver's own geometry, not from tower height or travel distance.
    """
    out = grid.copy()
    towers = []
    for c in range(grid.shape[1]):
        r = 0
        while r < grid.shape[0]:
            if grid[r, c] != RED:
                r += 1
                continue
            r0 = r
            while r + 1 < grid.shape[0] and grid[r + 1, c] == RED:
                r += 1
            r1 = r
            if r1 - r0 + 1 >= 3:
                towers.append((-(r1 - r0 + 1), c, r0, r1, c))
            r += 1
    towers.sort()
    _, _, r0, r1, tower_col = towers[0]
    receivers = _receiver_info(grid)
    activated = set()
    for r in range(r0, r1 + 1):
        for dc in (-1, 1):
            c = tower_col + dc
            while 0 <= c < grid.shape[1] and out[r, c] == BLACK:
                out[r, c] = RED
                c += dc
            hit = c
            for idx, (rr0, rr1, cc0, cc1, _color) in enumerate(receivers):
                if rr0 <= r <= rr1 and ((dc == -1 and hit == cc1) or (dc == 1 and hit == cc0)):
                    activated.add(idx)
    for idx in activated:
        r0, r1, c0, c1, color = receivers[idx]
        side = r1 - r0 + 1
        for c in range(c0, c1 + 1):
            for step in range(1, side + 1):
                nr = r0 - step
                if nr < 0 or out[nr, c] != BLACK:
                    break
                out[nr, c] = color
            for step in range(1, side + 1):
                nr = r1 + step
                if nr >= grid.shape[0] or out[nr, c] != BLACK:
                    break
                out[nr, c] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 3, 4, 5, 6, 7, 8, 9]
    for _ in range(400):
        h = rng.randint(16, 24)
        w = rng.randint(18, 26)
        grid = np.zeros((h, w), dtype=int)
        main_col = rng.randint(4, w - 5)
        main_h = rng.randint(4, 6)
        main_r0 = rng.randint(4, h - main_h - 5)
        grid[main_r0:main_r0 + main_h, main_col] = RED
        for _ in range(rng.randint(0, 2)):
            col = rng.choice([c for c in range(1, w - 1) if c != main_col])
            height = rng.randint(3, main_h - 1)
            r0 = rng.randint(1, h - height - 2)
            grid[r0:r0 + height, col] = RED
        for color in rng.sample(colors, rng.randint(2, 4)):
            side = rng.choice([2, 3])
            place_left = rng.choice([True, False])
            if place_left:
                if main_col - side - 2 < 1:
                    continue
                c0 = rng.randint(1, main_col - side - 2)
            else:
                if main_col + 2 > w - side - 2:
                    continue
                c0 = rng.randint(main_col + 2, w - side - 2)
            r0 = rng.randint(1, h - side - 2)
            block = grid[r0:r0 + side, c0:c0 + side]
            if np.any(block != BLACK):
                continue
            grid[r0:r0 + side, c0:c0 + side] = color
        if np.count_nonzero(grid == RED) >= main_h:
            return grid
    raise RuntimeError("failed to generate task 14/4")
