import random
from collections import deque

import numpy as np

BLACK = 0
GREEN = 3
PALETTE = [1, 2, 4, 5, 6, 7, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _arrow_direction(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    if len(set(rows)) == 2:
        counts = {r: rows.count(r) for r in set(rows)}
        long_row = max(counts, key=counts.get)
        short_row = min(counts, key=counts.get)
        if counts[long_row] == 3 and counts[short_row] == 1:
            return "up" if short_row < long_row else "down"
    if len(set(cols)) == 2:
        counts = {c: cols.count(c) for c in set(cols)}
        long_col = max(counts, key=counts.get)
        short_col = min(counts, key=counts.get)
        if counts[long_col] == 3 and counts[short_col] == 1:
            return "left" if short_col < long_col else "right"
    raise ValueError("invalid arrow")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored rectangle along its green arrow, resolving overlaps by rectangle area.

    Input structure:
    - Black background with several solid colored rectangles.
    - Every rectangle has one adjacent green L-shaped arrow made from a three-pixel stem and one bend pixel.
    - Arrow orientation encodes one of the four cardinal movement directions.

    Rule:
    - For each rectangle, move it in the arrow's direction by a distance equal to the rectangle's shorter side.
    - Remove all original rectangles and all green arrows.
    - Redraw the moved rectangles on the same canvas, applying smaller areas first and larger areas last so
      that larger rectangles cover smaller ones on overlapping target pixels.
    - A rectangle that ends up fully hidden simply has no visible pixels in the output.

    Key insight:
    - The arrow gives direction, the rectangle's own geometry gives the movement distance, and the rectangle
      area gives the visibility priority when targets overlap.
    """
    h, w = grid.shape
    rects = []
    green_comps = _components(grid == GREEN)
    for color in np.unique(grid):
        if color in (BLACK, GREEN):
            continue
        for comp in _components(grid == color):
            r0, r1, c0, c1 = _bbox(comp)
            area = len(comp)
            arrows = []
            for a_comp in green_comps:
                if any(
                    r0 - 1 <= rr <= r1 + 1 and c0 - 1 <= cc <= c1 + 1
                    for rr, cc in a_comp
                ):
                    arrows.append(a_comp)
            if len(arrows) != 1:
                continue
            direction = _arrow_direction(arrows[0])
            rects.append(
                {
                    "color": int(color),
                    "top": r0,
                    "left": c0,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                    "area": area,
                    "direction": direction,
                }
            )

    out = np.zeros_like(grid)
    rects.sort(key=lambda item: item["area"])
    for rect in rects:
        dist = min(rect["height"], rect["width"])
        dr, dc = {
            "up": (-dist, 0),
            "down": (dist, 0),
            "left": (0, -dist),
            "right": (0, dist),
        }[rect["direction"]]
        nr0 = rect["top"] + dr
        nc0 = rect["left"] + dc
        for rr in range(rect["height"]):
            for cc in range(rect["width"]):
                tr, tc = nr0 + rr, nc0 + cc
                if 0 <= tr < h and 0 <= tc < w:
                    out[tr, tc] = rect["color"]
    return out


def _arrow_cells(top, left, height, width, direction, rng):
    if direction == "up":
        row = top - 1
        start = left + rng.randint(0, max(0, width - 1))
        start = max(1, min(start, left + width - 1))
        start = min(start, left + width - 1)
        start = max(1, min(start, left + width - 1))
        cols = [start - 1, start, start + 1]
        cols = [max(0, min(c, left + width)) for c in cols]
        cols = sorted(set(cols))
        while len(cols) < 3:
            cols.append(cols[-1] + 1)
        cells = [(row, c) for c in cols[:3]]
        cells.append((row - 1, rng.choice((cols[0], cols[2]))))
        return cells
    if direction == "down":
        row = top + height
        start = left + rng.randint(0, max(0, width - 1))
        start = max(1, min(start, left + width - 1))
        cols = [start - 1, start, start + 1]
        cols = [max(0, min(c, left + width)) for c in cols]
        cols = sorted(set(cols))
        while len(cols) < 3:
            cols.append(cols[-1] + 1)
        cells = [(row, c) for c in cols[:3]]
        cells.append((row + 1, rng.choice((cols[0], cols[2]))))
        return cells
    if direction == "left":
        col = left - 1
        start = top + rng.randint(0, max(0, height - 1))
        start = max(1, min(start, top + height - 1))
        rows = [start - 1, start, start + 1]
        rows = [max(0, min(r, top + height)) for r in rows]
        rows = sorted(set(rows))
        while len(rows) < 3:
            rows.append(rows[-1] + 1)
        cells = [(r, col) for r in rows[:3]]
        cells.append((rng.choice((rows[0], rows[2])), col - 1))
        return cells
    col = left + width
    start = top + rng.randint(0, max(0, height - 1))
    start = max(1, min(start, top + height - 1))
    rows = [start - 1, start, start + 1]
    rows = [max(0, min(r, top + height)) for r in rows]
    rows = sorted(set(rows))
    while len(rows) < 3:
        rows.append(rows[-1] + 1)
    cells = [(r, col) for r in rows[:3]]
    cells.append((rng.choice((rows[0], rows[2])), col + 1))
    return cells


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        h = rng.randint(18, 28)
        w = rng.randint(18, 28)
        grid = np.zeros((h, w), dtype=int)
        used = np.zeros((h, w), dtype=bool)
        want_overlap = rng.choice([True, False])
        num_rects = rng.randint(2, 4)
        colors = rng.sample(PALETTE, num_rects)
        specs = []
        ok = True

        for color in colors:
            placed = False
            for _ in range(300):
                rh = rng.randint(2, 5)
                rw = rng.randint(2, 5)
                direction = rng.choice(["up", "down", "left", "right"])
                dist = min(rh, rw)
                if direction == "up":
                    if rh + dist + 2 >= h:
                        continue
                    top = rng.randint(dist + 2, h - rh - 1)
                    left = rng.randint(2, w - rw - 2)
                elif direction == "down":
                    if rh + dist + 2 >= h:
                        continue
                    top = rng.randint(1, h - rh - dist - 2)
                    left = rng.randint(2, w - rw - 2)
                elif direction == "left":
                    if rw + dist + 2 >= w:
                        continue
                    top = rng.randint(2, h - rh - 2)
                    left = rng.randint(dist + 2, w - rw - 1)
                else:
                    if rw + dist + 2 >= w:
                        continue
                    top = rng.randint(2, h - rh - 2)
                    left = rng.randint(1, w - rw - dist - 2)

                cells = [(r, c) for r in range(top, top + rh) for c in range(left, left + rw)]
                arrow = _arrow_cells(top, left, rh, rw, direction, rng)
                all_cells = cells + arrow
                if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                    continue
                rr0 = max(0, min(r for r, _ in all_cells) - 1)
                rr1 = min(h, max(r for r, _ in all_cells) + 2)
                cc0 = max(0, min(c for _, c in all_cells) - 1)
                cc1 = min(w, max(c for _, c in all_cells) + 2)
                if used[rr0:rr1, cc0:cc1].any():
                    continue
                for r, c in cells:
                    grid[r, c] = color
                for r, c in arrow:
                    grid[r, c] = GREEN
                used[rr0:rr1, cc0:cc1] = True
                specs.append((top, left, rh, rw, direction))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        targets = []
        areas = []
        for top, left, rh, rw, direction in specs:
            dist = min(rh, rw)
            dr, dc = {
                "up": (-dist, 0),
                "down": (dist, 0),
                "left": (0, -dist),
                "right": (0, dist),
            }[direction]
            targets.append((top + dr, left + dc, rh, rw))
            areas.append(rh * rw)

        overlap_found = False
        equal_overlap = False
        for i in range(len(targets)):
            ti, li, hi, wi = targets[i]
            set_i = {(r, c) for r in range(ti, ti + hi) for c in range(li, li + wi)}
            for j in range(i + 1, len(targets)):
                tj, lj, hj, wj = targets[j]
                set_j = {(r, c) for r in range(tj, tj + hj) for c in range(lj, lj + wj)}
                if set_i & set_j:
                    overlap_found = True
                    if areas[i] == areas[j]:
                        equal_overlap = True
        if equal_overlap:
            continue
        if want_overlap and not overlap_found:
            continue
        return grid
    raise RuntimeError("failed to generate input")
