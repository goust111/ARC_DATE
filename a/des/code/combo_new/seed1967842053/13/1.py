import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _legend_patterns(grid):
    wall_rows = np.where(grid == GREY)[0]
    first_wall = int(wall_rows.min()) if wall_rows.size else grid.shape[0]
    mask = (grid != BLACK) & (grid != GREY)
    patterns = {}
    legend_boxes = []
    for cells in _components(mask):
        r0, c0, r1, c1 = _bbox(cells)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if r1 < first_wall and 2 <= bh <= 3 and 2 <= bw <= 3:
            block = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            if np.all((block != BLACK) & (block != GREY)):
                patterns[int(block[0, 0])] = block
                legend_boxes.append((r0, c0, r1, c1))
    return patterns, legend_boxes


def _outside_reachable(mask_open):
    h, w = mask_open.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if mask_open[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask_open[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask_open[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _boundary_origin(cells, grey_mask):
    h, w = grey_mask.shape
    rows = []
    cols = []
    for r, c in cells:
        rows.append(r)
        cols.append(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grey_mask[nr, nc]:
                rows.append(nr)
                cols.append(nc)
    return min(rows), min(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey-enclosed region with the texture selected by its seed color.

    Input structure:
    The grid has black background, grey one-pixel closed outlines, and a legend
    band at the top made from separated 2x2, 2x3, 3x2, or 3x3 non-grey texture
    blocks. Each outlined interior contains exactly one non-black, non-grey seed
    pixel whose color equals the top-left index color of one legend block.

    Rule:
    Keep grey outlines fixed. Read every top legend block as a repeating texture
    keyed by its top-left color, erase those legend blocks to black, then flood
    from the border through non-grey cells to isolate enclosed interiors. For
    each interior, replace all its cells, including the seed, by the matching
    texture tiled from the top-left corner of that closed region's bounding box.

    Key insight:
    The seed is not a fill color; it is a pointer into the top legend, so the
    task is solved by extracting the legend textures before filling the regions.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    patterns, legend_boxes = _legend_patterns(grid)
    for r0, c0, r1, c1 in legend_boxes:
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    work = out.copy()
    grey_mask = work == GREY
    outside = _outside_reachable(~grey_mask)
    interior_mask = (~grey_mask) & (~outside)

    for cells in _components(interior_mask):
        seed_colors = [int(grid[r, c]) for r, c in cells if grid[r, c] not in (BLACK, GREY)]
        if not seed_colors:
            continue
        pattern = patterns.get(seed_colors[0])
        if pattern is None:
            continue
        pr, pc = pattern.shape
        origin_r, origin_c = _boundary_origin(cells, grey_mask)
        for r, c in cells:
            out[r, c] = pattern[(r - origin_r) % pr, (c - origin_c) % pc]
    return out


def _make_pattern(index_color, other_color):
    ph = random.randint(2, 3)
    pw = random.randint(2, 3)
    kind = random.choice(("checker", "hstripe", "vstripe"))
    arr = np.zeros((ph, pw), dtype=int)
    for r in range(ph):
        for c in range(pw):
            if kind == "checker":
                use_index = (r + c) % 2 == 0
            elif kind == "hstripe":
                use_index = r % 2 == 0
            else:
                use_index = c % 2 == 0
            arr[r, c] = index_color if use_index else other_color
    return arr


def _make_shape():
    h = random.randint(3, 6)
    w = random.randint(3, 7)
    mask = np.ones((h, w), dtype=bool)
    style = random.choice(("rect", "corner_notch", "side_notch", "stair"))
    if style == "corner_notch" and h >= 5 and w >= 5:
        nh = random.randint(1, min(2, h - 3))
        nw = random.randint(1, min(2, w - 3))
        corner = random.randrange(4)
        rs = slice(0, nh) if corner < 2 else slice(h - nh, h)
        cs = slice(0, nw) if corner in (0, 2) else slice(w - nw, w)
        mask[rs, cs] = False
    elif style == "side_notch" and h >= 5 and w >= 5:
        if random.choice((True, False)):
            nr = random.randint(1, h - 3)
            if random.choice((True, False)):
                mask[nr : nr + 2, 0] = False
            else:
                mask[nr : nr + 2, w - 1] = False
        else:
            nc = random.randint(1, w - 3)
            if random.choice((True, False)):
                mask[0, nc : nc + 2] = False
            else:
                mask[h - 1, nc : nc + 2] = False
    elif style == "stair" and h >= 4 and w >= 5:
        for r in range(h):
            cut = min(w - 3, r // 2)
            if random.choice((True, False)):
                mask[r, :cut] = False
            else:
                mask[r, w - cut :] = False
    return mask


def _wall_from_interior(interior):
    h, w = interior.shape
    local = np.zeros((h + 2, w + 2), dtype=int)
    local[1 : h + 1, 1 : w + 1][interior] = 1
    wall = np.zeros_like(local, dtype=bool)
    for r in range(1, h + 1):
        for c in range(1, w + 1):
            if not local[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if not local[nr, nc]:
                        wall[nr, nc] = True
    interior_cells = [(r + 1, c + 1) for r in range(h) for c in range(w) if interior[r, c]]
    wall_cells = [(r, c) for r in range(h + 2) for c in range(w + 2) if wall[r, c]]
    return h + 2, w + 2, interior_cells, wall_cells


def _place_legend(grid, patterns):
    h, w = grid.shape
    n = len(patterns)
    two_bands = n >= 3 and random.choice((True, False))
    groups = [list(range(n))]
    if two_bands:
        split = random.randint(1, n - 1)
        groups = [list(range(split)), list(range(split, n))]

    row = 0
    bottom = 0
    for group in groups:
        widths = [patterns[i].shape[1] for i in group]
        total = sum(widths) + max(0, len(group) - 1)
        if total > w:
            return None
        col = random.randint(0, w - total)
        band_h = max(patterns[i].shape[0] for i in group)
        for i in group:
            pat = patterns[i]
            ph, pw = pat.shape
            grid[row : row + ph, col : col + pw] = pat
            col += pw + 1
        bottom = max(bottom, row + band_h)
        row += band_h + 1
    return bottom


def generate_input() -> np.ndarray:
    for _ in range(200):
        height = random.randint(18, 30)
        width = random.randint(18, 30)
        grid = np.zeros((height, width), dtype=int)

        count = random.randint(2, 4)
        colors = random.sample(PALETTE, count * 2)
        index_colors = colors[:count]
        other_colors = colors[count:]
        patterns = [_make_pattern(index_colors[i], other_colors[i]) for i in range(count)]

        legend_bottom = _place_legend(grid, patterns)
        if legend_bottom is None or legend_bottom + 8 >= height:
            continue

        objects = list(range(count))
        random.shuffle(objects)
        placed = True
        for obj_index in objects:
            for _try in range(120):
                interior = _make_shape()
                fh, fw, interior_cells, wall_cells = _wall_from_interior(interior)
                if fh + 4 >= height - legend_bottom or fw + 2 >= width:
                    continue
                r0 = random.randint(legend_bottom + 1, height - fh - 2)
                c0 = random.randint(1, width - fw - 2)
                r_start = max(0, r0 - 1)
                r_stop = min(height, r0 + fh + 1)
                c_start = max(0, c0 - 1)
                c_stop = min(width, c0 + fw + 1)
                if np.any(grid[r_start:r_stop, c_start:c_stop] != BLACK):
                    continue

                for rr, cc in wall_cells:
                    grid[r0 + rr, c0 + cc] = GREY
                globals_inside = [(r0 + rr, c0 + cc) for rr, cc in interior_cells]
                sr, sc = random.choice(globals_inside)
                grid[sr, sc] = index_colors[obj_index]
                break
            else:
                placed = False
                break
        if placed:
            return grid
    raise RuntimeError("could not generate a valid ARC grid")

