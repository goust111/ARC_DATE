import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _same_color_components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _is_rect_border(cells):
    s = set(cells)
    min_r, max_r = min(r for r, _ in s), max(r for r, _ in s)
    min_c, max_c = min(c for _, c in s), max(c for _, c in s)
    if max_r - min_r < 2 or max_c - min_c < 2:
        return None
    border = {(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)
              if r in (min_r, max_r) or c in (min_c, max_c)}
    return (min_r, max_r, min_c, max_c) if s == border else None


def _find_legends(grid):
    h, w = grid.shape
    legends = {}
    for r in range(h - 2):
        for c in range(w - 2):
            if not (r <= 2 or c <= 2 or r + 2 >= h - 3 or c + 2 >= w - 3):
                continue
            block = grid[r:r + 3, c:c + 3]
            where9 = np.argwhere(block == 9)
            if len(where9) != 1:
                continue
            vals = [int(x) for x in block.ravel() if x not in (0, 9)]
            if not vals:
                continue
            color = max(set(vals), key=vals.count)
            if vals.count(color) < 5:
                continue
            ir, ic = where9[0]
            dr, dc = int(ir) - 1, int(ic) - 1
            for name, vec in DIRS.items():
                if vec == (dr, dc):
                    legends[color] = name
    return legends


def main(grid: np.ndarray) -> np.ndarray:
    """Stack yellow pixels inside colored regions using legend direction and marker count.

    Input structure: Colored rectangular regions contain yellow movable pixels and
    orange count markers. Edge legends with the matching region color encode one
    of the four orthogonal slide directions.
    Rule: For each region, orange marker count is the maximum yellow stack depth
    per lane. Yellow pixels slide toward the legend direction's far wall and any
    lane overflow beyond that count is erased. Orange markers disappear.
    Key insight: Direction comes from the external legend, while capacity comes
    from the region's internal orange count; both are needed to place the final
    yellow stacks.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    legends = _find_legends(grid)
    region_borders = []
    for color in legends:
        for comp in _same_color_components(grid, color):
            box = _is_rect_border(comp)
            if box and min(box[0], box[2]) > 2:
                region_borders.append((color, box))

    for color, (min_r, max_r, min_c, max_c) in region_borders:
        direction = legends[color]
        interior = [(r, c) for r in range(min_r + 1, max_r) for c in range(min_c + 1, max_c)]
        limit = sum(grid[r, c] == 7 for r, c in interior)
        yellow = {(r, c) for r, c in interior if grid[r, c] == 4}
        for r, c in interior:
            if out[r, c] in (4, 7):
                out[r, c] = 0
        if limit == 0:
            continue
        if direction in ("left", "right"):
            for r in range(min_r + 1, max_r):
                count = sum((r, c) in yellow for c in range(min_c + 1, max_c))
                keep = min(count, limit)
                if direction == "right":
                    cols = range(max_c - 1, min_c, -1)
                else:
                    cols = range(min_c + 1, max_c)
                for c in list(cols)[:keep]:
                    out[r, c] = 4
        else:
            for c in range(min_c + 1, max_c):
                count = sum((r, c) in yellow for r in range(min_r + 1, max_r))
                keep = min(count, limit)
                if direction == "down":
                    rows = range(max_r - 1, min_r, -1)
                else:
                    rows = range(min_r + 1, max_r)
                for r in list(rows)[:keep]:
                    out[r, c] = 4
    return out


def _draw_rect_border(grid, top, left, height, width, color):
    bottom, right = top + height - 1, left + width - 1
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color
    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color


def _draw_legend(grid, top, left, color, direction):
    grid[top:top + 3, left:left + 3] = color
    dr, dc = DIRS[direction]
    grid[top + 1 + dr, left + 1 + dc] = 9


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    regions = [(5, 1, 7, 6, 1), (5, 10, 7, 6, 2)]
    if random.random() < 0.5:
        regions.append((11, 5, 6, 7, 3))
    directions = random.sample(["up", "down", "left", "right"], len(regions))
    legend_positions = [(0, 0), (0, 5), (0, 10)]
    for idx, (top, left, height, width, color) in enumerate(regions):
        _draw_rect_border(grid, top, left, height, width, color)
        _draw_legend(grid, legend_positions[idx][0], legend_positions[idx][1], color, directions[idx])
        interior = [(r, c) for r in range(top + 1, top + height - 1) for c in range(left + 1, left + width - 1)]
        random.shuffle(interior)
        marker_count = random.randint(1, 3)
        for r, c in interior[:marker_count]:
            grid[r, c] = 7
        for r, c in interior[marker_count:marker_count + random.randint(5, 9)]:
            grid[r, c] = 4
    return grid

