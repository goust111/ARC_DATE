import random
from collections import deque

import numpy as np


_DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return colors[int(np.argmax(counts))]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _marker_direction(rr, cc, size):
    mid = size // 2
    if rr == 0 and cc == mid:
        return "U"
    if rr == size - 1 and cc == mid:
        return "D"
    if cc == 0 and rr == mid:
        return "L"
    if cc == size - 1 and rr == mid:
        return "R"
    return None


def _find_objects(grid, bg):
    objects = []
    for cells in _components(grid != bg):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        size_r = r1 - r0 + 1
        size_c = c1 - c0 + 1
        if size_r != size_c:
            continue
        size = size_r
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(block == bg):
            continue
        colors, counts = np.unique(block, return_counts=True)
        fill = colors[int(np.argmax(counts))]
        marker_cells = np.argwhere(block != fill)
        if len(marker_cells) != 1:
            continue
        rr, cc = map(int, marker_cells[0])
        direction = _marker_direction(rr, cc, size)
        if direction is None:
            continue
        objects.append({"r": r0, "c": c0, "size": size, "fill": fill, "dir": direction})
    return objects


def _can_step(occ, r, c, size, dr, dc):
    h, w = occ.shape
    nr, nc = r + dr, c + dc
    if nr < 0 or nc < 0 or nr + size > h or nc + size > w:
        return False
    return not occ[nr : nr + size, nc : nc + size].any()


def _render(grid):
    bg = _background(grid)
    h, w = grid.shape
    out = np.full_like(grid, bg)
    objects = _find_objects(grid, bg)
    order = (
        ("U", lambda obj: (obj["r"], obj["c"])),
        ("D", lambda obj: (-(obj["r"] + obj["size"] - 1), obj["c"])),
        ("L", lambda obj: (obj["c"], obj["r"])),
        ("R", lambda obj: (-(obj["c"] + obj["size"] - 1), obj["r"])),
    )

    for direction, key in order:
        occ = np.zeros((h, w), dtype=bool)
        group = sorted((obj for obj in objects if obj["dir"] == direction), key=key)
        dr, dc = _DIRS[direction]
        for obj in group:
            r, c, size = obj["r"], obj["c"], obj["size"]
            while _can_step(occ, r, c, size, dr, dc):
                r += dr
                c += dc
            out[r : r + size, c : c + size] = obj["fill"]
            occ[r : r + size, c : c + size] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each marked solid square to the edge indicated by its internal side marker, stack same-direction squares, and erase the marker.

    Input structure:
    The grid has a most-common background color and several separated equal-sized solid square blocks. Each block is filled with one non-background color and contains exactly one special marker pixel of another color at the center of its top, bottom, left, or right edge.
    Rule:
    For every block, read the marker position as the movement direction: top means up, bottom means down, left means left, and right means right. Clear the original positions, slide each whole block along that direction until it reaches the matching grid boundary, and for blocks moving in the same direction process the nearer block first so later overlapping paths stack directly behind it. At the final position, recolor the marker pixel back to the block fill color.
    Key insight:
    The direction is encoded inside each square by the asymmetric marker position, so the block itself supplies both the object and its sliding instruction.
    """
    return _render(np.asarray(grid))


def _fits(occ, r, c, size):
    h, w = occ.shape
    r0, r1 = max(0, r - 1), min(h, r + size + 1)
    c0, c1 = max(0, c - 1), min(w, c + size + 1)
    return not occ[r0:r1, c0:c1].any()


def _position_for(direction, h, w, size, placed):
    same = [obj for obj in placed if obj["dir"] == direction]
    reuse_lane = same and random.random() < 0.7
    if direction == "U":
        r = random.randint(1, h - size)
        c = same[-1]["c"] if reuse_lane else random.randint(0, w - size)
    elif direction == "D":
        r = random.randint(0, h - size - 1)
        c = same[-1]["c"] if reuse_lane else random.randint(0, w - size)
    elif direction == "L":
        r = same[-1]["r"] if reuse_lane else random.randint(0, h - size)
        c = random.randint(1, w - size)
    else:
        r = same[-1]["r"] if reuse_lane else random.randint(0, h - size)
        c = random.randint(0, w - size - 1)
    return r, c


def _draw_block(grid, r, c, size, fill, marker, direction):
    grid[r : r + size, c : c + size] = fill
    mid = size // 2
    if direction == "U":
        grid[r, c + mid] = marker
    elif direction == "D":
        grid[r + size - 1, c + mid] = marker
    elif direction == "L":
        grid[r + mid, c] = marker
    else:
        grid[r + mid, c + size - 1] = marker


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(16, 30)
        w = random.randint(16, 30)
        size = 5 if min(h, w) >= 24 and random.random() < 0.25 else 3
        capacity = min(8, max(4, (h * w) // (size * size * 5)))
        count = random.randint(5 if capacity >= 5 else 4, capacity)

        colors = list(range(10))
        bg = random.choice(colors)
        marker_choices = [color for color in colors if color != bg]
        marker = random.choice(marker_choices)
        fills = [color for color in colors if color not in (bg, marker)]
        random.shuffle(fills)

        directions = ["U", "D", "L", "R"]
        random.shuffle(directions)
        directions += [random.choice(["U", "D", "L", "R"]) for _ in range(count - 4)]

        grid = np.full((h, w), bg, dtype=int)
        occ = np.zeros((h, w), dtype=bool)
        placed = []
        ok = True
        for i, direction in enumerate(directions):
            for _attempt in range(200):
                r, c = _position_for(direction, h, w, size, placed)
                if _fits(occ, r, c, size):
                    fill = fills[i % len(fills)]
                    _draw_block(grid, r, c, size, fill, marker, direction)
                    occ[r : r + size, c : c + size] = True
                    placed.append({"r": r, "c": c, "size": size, "dir": direction})
                    break
            else:
                ok = False
                break
        if not ok:
            continue

        out = _render(grid)
        if np.count_nonzero(out != bg) == count * size * size:
            return grid

    h, w, size = 18, 18, 3
    bg, marker = 0, 1
    grid = np.full((h, w), bg, dtype=int)
    specs = [
        (2, 2, 2, "U"),
        (10, 4, 3, "D"),
        (6, 9, 4, "L"),
        (12, 12, 5, "R"),
        (7, 2, 6, "U"),
    ]
    for r, c, fill, direction in specs:
        _draw_block(grid, r, c, size, fill, marker, direction)
    return grid
