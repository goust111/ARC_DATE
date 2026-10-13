import random
from collections import deque

import numpy as np


BACKGROUND = 0


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Move each colored stamp to its same-colored address marker, rotating by marker direction.

    Input structure:
    The grid has a black background.  For each active color there is one small 4-connected
    multi-pixel stamp and one isolated single-pixel marker of the same color, separated from
    the stamp and from other colors.

    Rule:
    For each color, compare the marker position with the center of the stamp bounding box.
    A marker above the stamp keeps the stamp unrotated, a marker to the right rotates it
    clockwise 90 degrees, a marker below rotates it 180 degrees, and a marker to the left
    rotates it counterclockwise 90 degrees.  The rotated stamp is copied so its bounding-box
    center is aligned to the marker, while the original stamp and marker are cleared.

    Key insight:
    The single-pixel component is not the object to preserve; it is the address whose relative
    direction encodes the rotation to apply to the matching colored stamp.
    """
    grid = np.asarray(grid)
    out = np.zeros_like(grid)

    for color in np.unique(grid):
        if color == BACKGROUND:
            continue
        comps = _components(grid, color)
        stamps = [comp for comp in comps if len(comp) > 1]
        markers = [comp[0] for comp in comps if len(comp) == 1]
        if not stamps or not markers:
            continue

        stamp = max(stamps, key=len)
        marker_r, marker_c = markers[0]

        rows = [r for r, _ in stamp]
        cols = [c for _, c in stamp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0

        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in stamp:
            mask[r - r0, c - c0] = True

        dr = marker_r - center_r
        dc = marker_c - center_c
        if abs(dr) > abs(dc):
            rot = mask if dr < 0 else np.rot90(mask, 2)
        else:
            rot = np.rot90(mask, -1) if dc > 0 else np.rot90(mask, 1)

        rh, rw = rot.shape
        top = int(np.floor(marker_r - (rh - 1) / 2.0))
        left = int(np.floor(marker_c - (rw - 1) / 2.0))
        for rr, cc in np.argwhere(rot):
            tr = top + int(rr)
            tc = left + int(cc)
            if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                out[tr, tc] = color

    return out


def _is_connected(mask):
    coords = [tuple(x) for x in np.argwhere(mask)]
    if not coords:
        return False
    todo = [coords[0]]
    seen = {coords[0]}
    cells = set(coords)
    while todo:
        r, c = todo.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return len(seen) == len(cells)


def _good_shape(mask):
    if mask.sum() < 5 or mask.sum() > 6:
        return False
    if not (mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any()):
        return False
    if not _is_connected(mask):
        return False
    return not any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _random_shape():
    for _ in range(500):
        target = random.randint(5, 6)
        cells = {(1, 1)}
        while len(cells) < target:
            r, c = random.choice(tuple(cells))
            choices = []
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < 3 and 0 <= nc < 3:
                    choices.append((nr, nc))
            cells.add(random.choice(choices))
        mask = np.zeros((3, 3), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _good_shape(mask):
            return mask

    templates = [
        [[0, 1, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 1], [0, 1, 0]],
        [[0, 1, 1], [1, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [1, 1, 1], [0, 1, 0]],
    ]
    mask = np.array(random.choice(templates), dtype=bool)
    if random.choice((False, True)):
        mask = np.fliplr(mask)
    return np.rot90(mask, random.randrange(4))


def _reserved_box(r0, c0, h, w, rows, cols, pad=1):
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return None
    cells = set()
    for r in range(max(0, r0 - pad), min(rows, r0 + h + pad)):
        for c in range(max(0, c0 - pad), min(cols, c0 + w + pad)):
            cells.add((r, c))
    return cells


def _try_place(grid, blocked, color, direction):
    rows, cols = grid.shape
    shape = _random_shape()
    for _ in range(800):
        sr = random.randint(1, rows - 4)
        sc = random.randint(1, cols - 4)
        center_r = sr + 1
        center_c = sc + 1
        dist = random.randint(6, 10)
        jitter = random.choice((-1, 0, 1))

        if direction == "up":
            mr, mc = center_r - dist, center_c + jitter
        elif direction == "right":
            mr, mc = center_r + jitter, center_c + dist
        elif direction == "down":
            mr, mc = center_r + dist, center_c + jitter
        else:
            mr, mc = center_r + jitter, center_c - dist

        if not (1 <= mr < rows - 1 and 1 <= mc < cols - 1):
            continue
        if abs(mr - center_r) <= abs(mc - center_c) and direction in ("up", "down"):
            continue
        if abs(mc - center_c) <= abs(mr - center_r) and direction in ("left", "right"):
            continue

        stamp_cells = _reserved_box(sr, sc, 3, 3, rows, cols)
        marker_cells = _reserved_box(mr - 1, mc - 1, 3, 3, rows, cols)
        if stamp_cells is None or marker_cells is None:
            continue
        reserve = stamp_cells | marker_cells
        if reserve & blocked:
            continue

        for rr, cc in np.argwhere(shape):
            grid[sr + int(rr), sc + int(cc)] = color
        grid[mr, mc] = color
        blocked.update(reserve)
        return True
    return False


def generate_input() -> np.ndarray:
    rows = random.randint(24, 30)
    cols = random.randint(24, 30)
    count = random.randint(4, 5)
    colors = random.sample(range(1, 10), count)
    directions = ["up", "right", "down", "left"]
    random.shuffle(directions)
    if count > 4:
        directions.append(random.choice(("up", "right", "down", "left")))
    random.shuffle(directions)

    for _ in range(100):
        grid = np.zeros((rows, cols), dtype=int)
        blocked = set()
        ok = True
        for color, direction in zip(colors, directions):
            if not _try_place(grid, blocked, color, direction):
                ok = False
                break
        if ok:
            return grid

    return generate_input()
