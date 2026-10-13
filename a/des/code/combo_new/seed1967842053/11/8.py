import random
from collections import deque

import numpy as np


BLACK = 0


def _components_for_color(grid, color):
    rows, cols = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not seen[nr, nc]
                    and grid[nr, nc] == color
                ):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each colored segment toward its same-colored marker by its own length.

    Input structure:
    The background is black. Each non-black color appears as exactly one straight
    horizontal or vertical one-pixel-wide segment and one separated single-pixel
    marker of the same color. The marker lies beyond one end of its matching
    segment on the segment's axis, with black space between them.

    Rule:
    For every color, identify the multi-pixel component as the original segment
    and the isolated same-colored pixel as the marker. Determine which segment
    end points toward that marker, then extend from that end along the segment's
    axis. Add at most as many pixels as the original segment length, stopping
    early if the marker position is reached.

    Key insight:
    The marker supplies only the direction, while the segment's own pixel count
    supplies the growth distance.
    """
    out = np.array(grid, copy=True)
    for color in [int(x) for x in np.unique(grid) if int(x) != BLACK]:
        comps = _components_for_color(grid, color)
        if len(comps) < 2:
            continue

        segment = max(comps, key=len)
        markers = [pt for comp in comps if comp is not segment for pt in comp]
        if not markers or len(segment) < 2:
            continue
        marker = markers[0]

        rs = [r for r, _ in segment]
        cs = [c for _, c in segment]
        length = len(segment)

        if len(set(rs)) == 1:
            row = rs[0]
            left, right = min(cs), max(cs)
            mr, mc = marker
            if mr != row:
                continue
            if mc < left:
                start = (row, left)
                step = (0, -1)
                distance = left - mc
            elif mc > right:
                start = (row, right)
                step = (0, 1)
                distance = mc - right
            else:
                continue
        elif len(set(cs)) == 1:
            col = cs[0]
            top, bottom = min(rs), max(rs)
            mr, mc = marker
            if mc != col:
                continue
            if mr < top:
                start = (top, col)
                step = (-1, 0)
                distance = top - mr
            elif mr > bottom:
                start = (bottom, col)
                step = (1, 0)
                distance = mr - bottom
            else:
                continue
        else:
            continue

        grow = min(length, distance)
        sr, sc = start
        dr, dc = step
        for k in range(1, grow + 1):
            nr, nc = sr + dr * k, sc + dc * k
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = color

    return out


def _near_existing(cells, occupied, rows, cols):
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in occupied:
                    return True
    return False


def _place_one(rows, cols, occupied):
    horizontal = random.choice([True, False])
    length = random.randint(2, 5)
    toward_positive = random.choice([True, False])

    if random.choice([True, False]):
        distance = random.randint(2, length)
    else:
        distance = random.randint(length + 1, length + 5)

    if horizontal:
        row = random.randrange(1, rows - 1)
        if toward_positive:
            min_start = 1
            max_start = cols - length - distance - 1
            if max_start < min_start:
                return None
            left = random.randint(min_start, max_start)
            line = [(row, c) for c in range(left, left + length)]
            marker = (row, left + length - 1 + distance)
        else:
            min_start = distance + 1
            max_start = cols - length - 1
            if max_start < min_start:
                return None
            left = random.randint(min_start, max_start)
            line = [(row, c) for c in range(left, left + length)]
            marker = (row, left - distance)
    else:
        col = random.randrange(1, cols - 1)
        if toward_positive:
            min_start = 1
            max_start = rows - length - distance - 1
            if max_start < min_start:
                return None
            top = random.randint(min_start, max_start)
            line = [(r, col) for r in range(top, top + length)]
            marker = (top + length - 1 + distance, col)
        else:
            min_start = distance + 1
            max_start = rows - length - 1
            if max_start < min_start:
                return None
            top = random.randint(min_start, max_start)
            line = [(r, col) for r in range(top, top + length)]
            marker = (top - distance, col)

    cells = line + [marker]
    if any(cell in occupied for cell in cells):
        return None
    if _near_existing(cells, occupied, rows, cols):
        return None
    return line, marker


def generate_input() -> np.ndarray:
    rows = random.randint(12, 30)
    cols = random.randint(12, 30)
    grid = np.zeros((rows, cols), dtype=int)

    max_objects = min(6, rows * cols // 35, 9)
    count = random.randint(2, max(2, max_objects))
    colors = random.sample(range(1, 10), count)
    occupied = set()

    for color in colors:
        placed = None
        for _ in range(300):
            placed = _place_one(rows, cols, occupied)
            if placed is not None:
                break
        if placed is None:
            continue
        line, marker = placed
        for r, c in line:
            grid[r, c] = color
            occupied.add((r, c))
        grid[marker] = color
        occupied.add(marker)

    if len(set(int(x) for x in np.unique(grid) if int(x) != BLACK)) >= 2:
        return grid
    return generate_input()
