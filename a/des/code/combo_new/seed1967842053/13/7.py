import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components_of_grey(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != GREY:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == GREY:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _container_from_component(grid, cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    present = {
        "up": bool(np.all(grid[r0, c0 : c1 + 1] == GREY)),
        "down": bool(np.all(grid[r1, c0 : c1 + 1] == GREY)),
        "left": bool(np.all(grid[r0 : r1 + 1, c0] == GREY)),
        "right": bool(np.all(grid[r0 : r1 + 1, c1] == GREY)),
    }
    missing = [side for side, ok in present.items() if not ok]
    if len(missing) != 1:
        return None
    opening = missing[0]

    expected = set()
    if opening != "up":
        expected.update((r0, c) for c in range(c0, c1 + 1))
    if opening != "down":
        expected.update((r1, c) for c in range(c0, c1 + 1))
    if opening != "left":
        expected.update((r, c0) for r in range(r0, r1 + 1))
    if opening != "right":
        expected.update((r, c1) for r in range(r0, r1 + 1))
    if set(cells) != expected:
        return None
    return r0, r1, c0, c1, opening


def _interior(r0, r1, c0, c1, opening):
    if opening == "up":
        return list(range(r0, r1)), list(range(c0 + 1, c1))
    if opening == "down":
        return list(range(r0 + 1, r1 + 1)), list(range(c0 + 1, c1))
    if opening == "left":
        return list(range(r0 + 1, r1)), list(range(c0, c1))
    return list(range(r0 + 1, r1)), list(range(c0 + 1, c1 + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """Repack each open grey container into sorted compact color columns.

    Input structure:
    The grid has a black background. Grey pixels form one or more rectangular
    U-shaped containers: three sides of a rectangle are present and exactly one
    side is open. Non-black, non-grey pixels are scattered inside each container.

    Rule:
    For every grey U-shape, detect which rectangle side is missing. Collect the
    colored interior pixels and group them by color. Clear the interior to black,
    then sort the color groups by their original mean coordinate on the axis
    perpendicular to the opening direction. Each group is written as one compact
    column parallel to the opening direction, starting at the closed wall opposite
    the opening. The grey frame is preserved.

    Key insight:
    The missing side of the grey rectangle encodes the gravity direction; colors
    keep only their relative cross-axis order while their pixels fall into tight
    stacks against the closed end of the container.
    """
    out = np.array(grid, copy=True)
    for cells in _components_of_grey(grid):
        container = _container_from_component(grid, cells)
        if container is None:
            continue
        r0, r1, c0, c1, opening = container
        rows, cols = _interior(r0, r1, c0, c1, opening)
        if not rows or not cols:
            continue

        groups = {}
        first_seen = {}
        for r in rows:
            for c in cols:
                value = int(grid[r, c])
                if value in (BLACK, GREY):
                    continue
                if value not in groups:
                    groups[value] = []
                    first_seen[value] = (r, c)
                groups[value].append((r, c))

        out[np.ix_(rows, cols)] = BLACK
        if not groups:
            continue

        vertical = opening in ("up", "down")

        def sort_key(item):
            color, pts = item
            if vertical:
                perp = [c for _, c in pts]
                para = [r for r, _ in pts]
            else:
                perp = [r for r, _ in pts]
                para = [c for _, c in pts]
            first = first_seen[color]
            return (sum(perp) / len(perp), min(perp), sum(para) / len(para), first[0], first[1])

        ordered = sorted(groups.items(), key=sort_key)
        if vertical:
            lanes = cols
            for idx, (color, pts) in enumerate(ordered[: len(lanes)]):
                col = lanes[idx]
                fill_rows = list(reversed(rows)) if opening == "up" else rows
                for r in fill_rows[: len(pts)]:
                    out[r, col] = color
        else:
            lanes = rows
            for idx, (color, pts) in enumerate(ordered[: len(lanes)]):
                row = lanes[idx]
                fill_cols = list(reversed(cols)) if opening == "left" else cols
                for c in fill_cols[: len(pts)]:
                    out[row, c] = color
    return out


def _draw_container(grid, r0, c0, height, width, opening):
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if opening != "up":
        grid[r0, c0 : c1 + 1] = GREY
    if opening != "down":
        grid[r1, c0 : c1 + 1] = GREY
    if opening != "left":
        grid[r0 : r1 + 1, c0] = GREY
    if opening != "right":
        grid[r0 : r1 + 1, c1] = GREY
    return r1, c1


def _place_colors(grid, r0, r1, c0, c1, opening):
    rows, cols = _interior(r0, r1, c0, c1, opening)
    vertical = opening in ("up", "down")
    lane_count = len(cols) if vertical else len(rows)
    run_length = len(rows) if vertical else len(cols)
    color_count = random.randint(1, min(4, lane_count))
    colors = random.sample(COLORS, color_count)
    lanes = random.sample(cols if vertical else rows, color_count)

    for color, lane in zip(colors, lanes):
        count = random.randint(1, run_length)
        if vertical:
            chosen_rows = random.sample(rows, count)
            for r in chosen_rows:
                grid[r, lane] = color
        else:
            chosen_cols = random.sample(cols, count)
            for c in chosen_cols:
                grid[lane, c] = color


def generate_input() -> np.ndarray:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    target_count = random.randint(1, 4)
    placed = 0

    for _ in range(target_count):
        for _attempt in range(200):
            opening = random.choice(["up", "down", "left", "right"])
            height = random.randint(4, 8)
            width = random.randint(4, 8)
            if height + 2 > h or width + 2 > w:
                continue
            r0 = random.randint(1, h - height - 1)
            c0 = random.randint(1, w - width - 1)
            rr0, rr1 = max(0, r0 - 1), min(h, r0 + height + 1)
            cc0, cc1 = max(0, c0 - 1), min(w, c0 + width + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue

            r1, c1 = _draw_container(grid, r0, c0, height, width, opening)
            _place_colors(grid, r0, r1, c0, c1, opening)
            occupied[rr0:rr1, cc0:cc1] = True
            placed += 1
            break

    if placed == 0:
        opening = random.choice(["up", "down", "left", "right"])
        height = random.randint(5, min(8, h - 2))
        width = random.randint(5, min(8, w - 2))
        r0 = random.randint(1, h - height - 1)
        c0 = random.randint(1, w - width - 1)
        r1, c1 = _draw_container(grid, r0, c0, height, width, opening)
        _place_colors(grid, r0, r1, c0, c1, opening)

    return grid
