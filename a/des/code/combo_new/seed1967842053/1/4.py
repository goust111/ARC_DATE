import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _border_color(patch):
    vals, counts = np.unique(patch[patch != 0], return_counts=True)
    return int(vals[np.argmax(counts)])


def _layer_count(patch):
    color = _border_color(patch)
    mask = (patch != 0) & (patch != color)
    if not mask.any():
        return 0

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top_gap = rows.min()
    bottom_gap = patch.shape[0] - 1 - rows.max()
    left_gap = cols.min()
    right_gap = patch.shape[1] - 1 - cols.max()
    side = int(np.argmax([top_gap, bottom_gap, left_gap, right_gap]))

    if side in (0, 1):
        return int(mask.any(axis=1).sum())
    return int(mask.any(axis=0).sum())


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each container by the number of internal fill layers.

    Input structure:
    The grid has black background and several separated nonzero components. Each component is a
    one-pixel-wide U-shaped, L-shaped, or right-angle-slot container in one border color, with one
    to three visibly distinct stripe layers packed inside from the closed end toward its opening.

    Rule:
    For every connected container component, count the colored internal fill layers, rotate the
    whole component clockwise by N * 90 degrees around its bounding-rectangle center. If the
    rotated local rectangle extends beyond the old rectangle, the output canvas grows as needed.

    Key insight:
    The visible count of fill stripes is an instruction: one, two, or three layers encode quarter,
    half, or three-quarter clockwise turns of the entire local object.
    """
    grid = np.array(grid, dtype=int, copy=False)
    placements = []
    min_r = min_c = 0
    max_r, max_c = grid.shape
    for cells in _components(grid):
        r0, c0, r1, c1 = _bbox(cells)
        patch = grid[r0:r1, c0:c1]
        turns = _layer_count(patch) % 4
        rot = np.rot90(patch, -turns)
        ph, pw = patch.shape
        rh, rw = rot.shape
        nr0 = r0 + (ph - rh) // 2
        nc0 = c0 + (pw - rw) // 2
        placements.append((nr0, nc0, rot))
        min_r = min(min_r, nr0)
        min_c = min(min_c, nc0)
        max_r = max(max_r, nr0 + rh)
        max_c = max(max_c, nc0 + rw)

    off_r = -min_r
    off_c = -min_c
    out = np.zeros((max_r + off_r, max_c + off_c), dtype=grid.dtype)
    for r0, c0, rot in placements:
        rh, rw = rot.shape
        region = out[r0 + off_r:r0 + off_r + rh, c0 + off_c:c0 + off_c + rw]
        region[rot != 0] = rot[rot != 0]
    return out


def _container_patch(width, height, opening, layers, border, fills, shape):
    p = np.zeros((height, width), dtype=int)
    if opening == "up":
        p[-1, :] = border
        p[:, 0] = border
        p[:, -1] = border
        for i in range(layers):
            p[height - 2 - i, 1:width - 1] = fills[i]
        if shape == "l":
            p[:max(1, height // 3), -1] = 0
        elif shape == "slot":
            p[0, width // 2:width - 1] = border
    elif opening == "down":
        p[0, :] = border
        p[:, 0] = border
        p[:, -1] = border
        for i in range(layers):
            p[1 + i, 1:width - 1] = fills[i]
        if shape == "l":
            p[height - max(1, height // 3):, -1] = 0
        elif shape == "slot":
            p[-1, 1:width // 2 + 1] = border
    elif opening == "left":
        p[:, -1] = border
        p[0, :] = border
        p[-1, :] = border
        for i in range(layers):
            p[1:height - 1, width - 2 - i] = fills[i]
        if shape == "l":
            p[-1, :max(1, width // 3)] = 0
        elif shape == "slot":
            p[1:height // 2 + 1, 0] = border
    else:
        p[:, 0] = border
        p[0, :] = border
        p[-1, :] = border
        for i in range(layers):
            p[1:height - 1, 1 + i] = fills[i]
        if shape == "l":
            p[-1, width - max(1, width // 3):] = 0
        elif shape == "slot":
            p[height // 2:height - 1, -1] = border
    return p


def _fits(used, r, c, h, w, margin=1):
    h0, w0 = used.shape
    if r < margin or c < margin or r + h + margin > h0 or c + w + margin > w0:
        return False
    area = used[r - margin:r + h + margin, c - margin:c + w + margin]
    return not area.any()


def generate_input() -> np.ndarray:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    grid = np.zeros((h, w), dtype=int)
    used = np.zeros((h, w), dtype=bool)
    colors = list(range(1, 10))
    random.shuffle(colors)
    openings = ["up", "down", "left", "right"]
    shapes = ["u", "l", "slot"]
    count = random.randint(1, 3)
    layer_plan = random.sample([1, 2, 3], count)
    made = 0
    attempts = 0

    while made < count and attempts < 250:
        attempts += 1
        opening = openings[(made + random.randint(0, 3)) % 4]
        shape = random.choice(shapes)
        layers = layer_plan[made]
        random.shuffle(colors)
        border = colors[0]
        fills = colors[1:1 + layers]

        if opening in ("up", "down"):
            ph = random.randint(layers + 3, layers + 5)
            pw = random.randint(4, 7)
        else:
            ph = random.randint(4, 7)
            pw = random.randint(layers + 3, layers + 5)
        if (ph - pw) % 2:
            if opening in ("up", "down") and pw < 7:
                pw += 1
            elif opening in ("left", "right") and ph < 7:
                ph += 1
            else:
                ph -= 1
        if ph + 2 > h or pw + 2 > w:
            continue

        patch = _container_patch(pw, ph, opening, layers, border, fills, shape)
        rh, rw = patch.shape if layers % 2 == 0 else (patch.shape[1], patch.shape[0])
        dr = (patch.shape[0] - rh) // 2
        dc = (patch.shape[1] - rw) // 2
        min_dr, min_dc = min(0, dr), min(0, dc)
        max_dr, max_dc = max(patch.shape[0], dr + rh), max(patch.shape[1], dc + rw)
        need_h = max_dr - min_dr
        need_w = max_dc - min_dc
        if need_h + 2 > h or need_w + 2 > w:
            continue

        base_r = random.randint(1, h - need_h - 1)
        base_c = random.randint(1, w - need_w - 1)
        r = base_r - min_dr
        c = base_c - min_dc
        if not _fits(used, base_r, base_c, need_h, need_w):
            continue

        grid[r:r + patch.shape[0], c:c + patch.shape[1]][patch != 0] = patch[patch != 0]
        used[base_r - 1:base_r + need_h + 1, base_c - 1:base_c + need_w + 1] = True
        made += 1

    if made == 0:
        patch = _container_patch(6, 6, "up", 2, 1, [2, 3], "u")
        grid[2:8, 2:8][patch != 0] = patch[patch != 0]
    return grid
