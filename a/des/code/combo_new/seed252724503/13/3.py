import random
from collections import deque

import numpy as np

BLACK = 0
GREY = 5


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


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
                for nr, nc in _neighbors(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _find_arrow_direction(grid: np.ndarray, cells):
    h, w = grid.shape
    s = set(cells)
    head = None
    for r, c in cells:
        deg = sum((nr, nc) in s for nr, nc in _neighbors(r, c, h, w))
        if deg == 2:
            head = (r, c)
            break
    assert head is not None, "expected an L-triomino arrow corner"
    hr, hc = head
    present = {(r - hr, c - hc) for r, c in cells if (r, c) != head}
    candidates = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if (dr, dc) not in present:
            candidates.append((dr, dc))
    for dr, dc in candidates:
        r, c = hr + dr, hc + dc
        while 0 <= r < h and 0 <= c < w:
            if grid[r, c] == GREY:
                return head, (dr, dc)
            r += dr
            c += dc
    return head, candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey target hit by each L-arrow and extend the arrow into a ray.

    Input structure:
    - Black background with several grey 4-connected target objects.
    - Colored arrow markers are 3-pixel L triominoes, one color per arrow.
    - Along one valid cardinal ray from each arrow's corner pixel, the first grey object is its unique target.

    Rule:
    - Detect each arrow's pointing direction from its L shape by choosing the empty cardinal branch whose ray reaches a grey object.
    - Recolor that first-hit grey object to the arrow color.
    - Erase the original arrow cells, then draw a 1-pixel ray of the same color from the arrow head to the grid edge, leaving target-object pixels untouched where the ray crosses them.

    Key insight:
    - Each L marker both selects a direction and supplies the repaint color, so the same control object triggers two linked effects: object recoloring and ray extension.
    """
    out = grid.copy()
    grey_parts = _components(grid == GREY)
    object_id = {}
    for idx, cells in enumerate(grey_parts):
        for cell in cells:
            object_id[cell] = idx
    for color in (int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY)):
        for cells in _components(grid == color):
            head, (dr, dc) = _find_arrow_direction(grid, cells)
            r, c = head[0] + dr, head[1] + dc
            target = None
            while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                if grid[r, c] == GREY:
                    target = object_id[(r, c)]
                    break
                r += dr
                c += dc
            if target is not None:
                for tr, tc in grey_parts[target]:
                    out[tr, tc] = color
            for ar, ac in cells:
                out[ar, ac] = BLACK
            r, c = head
            while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                if target is None or (r, c) not in grey_parts[target]:
                    if grid[r, c] != GREY:
                        out[r, c] = color
                r += dr
                c += dc
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    arrow_shapes = [
        ([(0, 0), (1, 0), (0, 1)], [(-1, 0), (0, -1)]),
        ([(0, 0), (1, 0), (0, -1)], [(-1, 0), (0, 1)]),
        ([(0, 0), (-1, 0), (0, 1)], [(1, 0), (0, -1)]),
        ([(0, 0), (-1, 0), (0, -1)], [(1, 0), (0, 1)]),
    ]
    for _ in range(400):
        h = rng.randint(16, 24)
        w = rng.randint(16, 24)
        grid = np.zeros((h, w), dtype=int)
        n = rng.randint(2, 4)
        placed = set()
        ok = True
        for color in rng.sample(colors, n):
            sh_cells, cand_dirs = rng.choice(arrow_shapes)
            actual_dir = rng.choice(cand_dirs)
            for _ in range(100):
                hr = rng.randint(2, h - 3)
                hc = rng.randint(2, w - 3)
                abs_cells = [(hr + dr, hc + dc) for dr, dc in sh_cells]
                if any(cell in placed for cell in abs_cells):
                    continue
                ray_len = rng.randint(3, 6)
                if actual_dir[0] == -1:
                    obj_h, obj_w = rng.randint(2, 3), rng.randint(2, 4)
                    r0 = hr - ray_len - obj_h + 1
                    c0 = hc - rng.randint(0, obj_w - 1)
                elif actual_dir[0] == 1:
                    obj_h, obj_w = rng.randint(2, 3), rng.randint(2, 4)
                    r0 = hr + ray_len
                    c0 = hc - rng.randint(0, obj_w - 1)
                elif actual_dir[1] == -1:
                    obj_h, obj_w = rng.randint(2, 4), rng.randint(2, 3)
                    r0 = hr - rng.randint(0, obj_h - 1)
                    c0 = hc - ray_len - obj_w + 1
                else:
                    obj_h, obj_w = rng.randint(2, 4), rng.randint(2, 3)
                    r0 = hr - rng.randint(0, obj_h - 1)
                    c0 = hc + ray_len
                if not (1 <= r0 and 1 <= c0 and r0 + obj_h < h - 1 and c0 + obj_w < w - 1):
                    continue
                obj_cells = [(r, c) for r in range(r0, r0 + obj_h) for c in range(c0, c0 + obj_w)]
                if any(cell in placed for cell in obj_cells):
                    continue
                blocked_other = False
                for dr, dc in cand_dirs:
                    if (dr, dc) == actual_dir:
                        continue
                    r, c = hr + dr, hc + dc
                    while 0 <= r < h and 0 <= c < w:
                        if (r, c) in obj_cells:
                            blocked_other = True
                            break
                        r += dr
                        c += dc
                if blocked_other:
                    continue
                for r, c in obj_cells:
                    grid[r, c] = GREY
                    placed.add((r, c))
                for r, c in abs_cells:
                    grid[r, c] = color
                    placed.add((r, c))
                break
            else:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate task 13/3")
