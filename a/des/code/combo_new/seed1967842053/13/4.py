import random
from collections import deque

import numpy as np


def _dominant_color(row):
    counts = {}
    best = int(row[0])
    best_count = 0
    for value in row:
        color = int(value)
        count = counts.get(color, 0) + 1
        counts[color] = count
        if count > best_count:
            best = color
            best_count = count
    return best


def _stripe_bands(grid):
    row_colors = [_dominant_color(grid[r]) for r in range(grid.shape[0])]
    bands = []
    start = 0
    for r in range(1, len(row_colors) + 1):
        if r == len(row_colors) or row_colors[r] != row_colors[start]:
            bands.append((start, r, row_colors[start]))
            start = r
    return bands


def _foreign_components(grid, band_start, band_end, stripe_color):
    h, w = band_end - band_start, grid.shape[1]
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for rr in range(h):
        for c in range(w):
            if seen[rr, c] or int(grid[band_start + rr, c]) == stripe_color:
                continue
            color = int(grid[band_start + rr, c])
            q = deque([(rr, c)])
            seen[rr, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if int(grid[band_start + nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            components.append(
                {
                    "color": color,
                    "top": band_start + min(rows),
                    "bottom": band_start + max(rows) + 1,
                    "left": min(cols),
                    "right": max(cols) + 1,
                    "offset": min(rows),
                }
            )
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Repair invaded horizontal stripes and mark each invader's source stripe with the repaired color.

    Input structure:
    The grid is filled by complete horizontal stripes, each occupying one or more consecutive rows and
    each having a distinct stripe color. Small solid rectangular invasion blocks sit fully inside single
    stripes; every block uses the color of another stripe and is visibly different from the stripe it covers.
    Blocks are separated and do not cross stripe boundaries.

    Rule:
    Infer each stripe's original color from the dominant visible color of its rows. For every foreign
    rectangular block inside a stripe, restore that block to the invaded stripe color. Then find the source
    stripe whose original color matches the block color, and recolor the same column range with the same
    height and relative vertical offset inside that source stripe to the invaded stripe color.

    Key insight:
    The visible patch is not only damage to repair; its color identifies a source stripe, so the output is a
    paired stripe-color exchange at the same local rectangle position.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    bands = _stripe_bands(grid)
    by_color = {color: (start, end) for start, end, color in bands}

    actions = []
    for start, end, stripe_color in bands:
        for comp in _foreign_components(grid, start, end, stripe_color):
            if comp["color"] not in by_color:
                continue
            actions.append((start, end, stripe_color, comp))

    for _start, _end, stripe_color, comp in actions:
        out[comp["top"] : comp["bottom"], comp["left"] : comp["right"]] = stripe_color

    for _start, _end, stripe_color, comp in actions:
        src_start, src_end = by_color[comp["color"]]
        height = comp["bottom"] - comp["top"]
        src_height = src_end - src_start
        if height > src_height:
            height = src_height
        offset = comp["offset"]
        if offset + height > src_height:
            offset = max(0, src_height - height)
        out[src_start + offset : src_start + offset + height, comp["left"] : comp["right"]] = stripe_color

    return out


def _rects_overlap(a, b, margin=0):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return ar0 - margin < br1 and br0 - margin < ar1 and ac0 - margin < bc1 and bc0 - margin < ac1


def generate_input() -> np.ndarray:
    stripe_count = random.randint(3, 5)
    heights = [random.randint(4, 6) for _ in range(stripe_count)]
    width = random.randint(12, 30)
    colors = random.sample(range(10), stripe_count)

    starts = []
    row = 0
    for height in heights:
        starts.append(row)
        row += height

    grid = np.zeros((sum(heights), width), dtype=int)
    for start, height, color in zip(starts, heights, colors):
        grid[start : start + height, :] = color

    patch_goal = random.randint(1, min(6, stripe_count * 2))
    reserved = []
    foreign_width_by_row = [0 for _ in range(grid.shape[0])]
    patch_count = 0

    for _ in range(patch_goal):
        placed = False
        for _attempt in range(200):
            dst = random.randrange(stripe_count)
            src_choices = [i for i in range(stripe_count) if i != dst]
            src = random.choice(src_choices)
            max_height = min(4, heights[dst], heights[src])
            patch_h = random.randint(2, max_height)
            max_offset = min(heights[dst], heights[src]) - patch_h
            offset = random.randint(0, max_offset)
            patch_w = random.randint(2, 4)
            left = random.randint(0, width - patch_w)
            right = left + patch_w

            dst_rect = (starts[dst] + offset, starts[dst] + offset + patch_h, left, right)
            src_rect = (starts[src] + offset, starts[src] + offset + patch_h, left, right)
            if any(_rects_overlap(dst_rect, rect, margin=1) for rect in reserved):
                continue
            if any(_rects_overlap(src_rect, rect, margin=0) for rect in reserved):
                continue
            if any(foreign_width_by_row[r] + patch_w >= width // 2 for r in range(dst_rect[0], dst_rect[1])):
                continue

            grid[dst_rect[0] : dst_rect[1], left:right] = colors[src]
            reserved.append(dst_rect)
            reserved.append(src_rect)
            for r in range(dst_rect[0], dst_rect[1]):
                foreign_width_by_row[r] += patch_w
            patch_count += 1
            placed = True
            break
        if not placed and patch_count == 0:
            continue

    if patch_count == 0:
        dst, src = 0, 1
        patch_h = 2
        patch_w = 2
        offset = 0
        left = 0
        grid[starts[dst] + offset : starts[dst] + offset + patch_h, left : left + patch_w] = colors[src]

    return grid
