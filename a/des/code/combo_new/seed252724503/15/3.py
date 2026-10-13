import random
from collections import deque

import numpy as np


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each 3x3 notched block opposite its notch by the number of same-color axis markers, then erase the markers.

    Input structure:
    - The grid has a dominant background color.
    - Several monochrome 3x3 blocks each contain exactly one missing edge-center pixel, so each block has area 8.
    - Additional isolated same-color singleton pixels lie on the controlling row or column of a matching block.

    Rule:
    - For each 3x3 block, detect which edge-center cell is missing.
    - Move the block in the opposite direction: left notch -> right, right notch -> left, top notch -> down, bottom notch -> up.
    - The travel distance equals the number of same-color singleton markers on the block's center row for horizontal motion or center column for vertical motion.
    - Remove all singleton markers from the output.

    Key insight:
    - The notch sets the motion direction, while the detached same-color markers on the aligned row or column encode the translation distance.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    out = np.full_like(g, bg)

    for color in [int(v) for v in np.unique(g) if v != bg]:
        comps = _components(g == color)
        blocks = []
        markers = set()
        for cells in comps:
            r0, c0, r1, c1 = _bbox(cells)
            if len(cells) == 8 and r1 - r0 == 2 and c1 - c0 == 2:
                blocks.append(cells)
            elif len(cells) == 1:
                markers.add(cells[0])
        for cells in blocks:
            r0, c0, _, _ = _bbox(cells)
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in cells:
                mask[r - r0, c - c0] = True
            missing = tuple(np.argwhere(~mask)[0])
            if missing == (1, 0):
                dr, dc = 0, 1
                count = sum(1 for r, c in markers if r == r0 + 1)
            elif missing == (1, 2):
                dr, dc = 0, -1
                count = sum(1 for r, c in markers if r == r0 + 1)
            elif missing == (0, 1):
                dr, dc = 1, 0
                count = sum(1 for r, c in markers if c == c0 + 1)
            else:
                dr, dc = -1, 0
                count = sum(1 for r, c in markers if c == c0 + 1)
            for r, c in cells:
                out[r + dr * count, c + dc * count] = color
    return out


def _place_block(grid, occ, color, top, left, missing):
    for rr in range(3):
        for cc in range(3):
            if (rr, cc) == missing:
                continue
            grid[top + rr, left + cc] = color
            occ[top + rr, left + cc] = True


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(14, 22), random.randint(14, 22)
        bg = random.choice([0, 1, 7])
        grid = np.full((h, w), bg, dtype=int)
        occ = np.zeros((h, w), dtype=bool)
        colors = random.sample([c for c in range(10) if c != bg], random.randint(2, 4))
        ok = True
        for color in colors:
            direction = random.choice(["left", "right", "up", "down"])
            n = random.randint(1, 4)
            if direction == "left":
                missing = (1, 0)
                row = random.randint(2, h - 3)
                left = random.randint(1, w - 4 - n)
                top = row - 1
                target_left = left + n
                if occ[top : top + 3, left : target_left + 3].any():
                    ok = False
                    break
                _place_block(grid, occ, color, top, left, missing)
                choices = [c for c in range(w) if c < left - 1 or c > target_left + 3]
                if len(choices) < n:
                    ok = False
                    break
                for c in random.sample(choices, n):
                    if occ[row, c]:
                        ok = False
                        break
                    grid[row, c] = color
                    occ[row, c] = True
            elif direction == "right":
                missing = (1, 2)
                row = random.randint(2, h - 3)
                left = random.randint(n + 1, w - 4)
                top = row - 1
                target_left = left - n
                if occ[top : top + 3, target_left : left + 3].any():
                    ok = False
                    break
                _place_block(grid, occ, color, top, left, missing)
                choices = [c for c in range(w) if c < target_left - 2 or c > left + 3]
                if len(choices) < n:
                    ok = False
                    break
                for c in random.sample(choices, n):
                    if occ[row, c]:
                        ok = False
                        break
                    grid[row, c] = color
                    occ[row, c] = True
            elif direction == "up":
                missing = (0, 1)
                col = random.randint(2, w - 3)
                top = random.randint(1, h - 4 - n)
                left = col - 1
                target_top = top + n
                if occ[top : target_top + 3, left : left + 3].any():
                    ok = False
                    break
                _place_block(grid, occ, color, top, left, missing)
                choices = [r for r in range(h) if r < top - 1 or r > target_top + 3]
                if len(choices) < n:
                    ok = False
                    break
                for r in random.sample(choices, n):
                    if occ[r, col]:
                        ok = False
                        break
                    grid[r, col] = color
                    occ[r, col] = True
            else:
                missing = (2, 1)
                col = random.randint(2, w - 3)
                top = random.randint(n + 1, h - 4)
                left = col - 1
                target_top = top - n
                if occ[target_top : top + 3, left : left + 3].any():
                    ok = False
                    break
                _place_block(grid, occ, color, top, left, missing)
                choices = [r for r in range(h) if r < target_top - 2 or r > top + 3]
                if len(choices) < n:
                    ok = False
                    break
                for r in random.sample(choices, n):
                    if occ[r, col]:
                        ok = False
                        break
                    grid[r, col] = color
                    occ[r, col] = True
            if not ok:
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
