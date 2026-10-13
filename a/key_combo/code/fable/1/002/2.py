import random

import numpy as np


BG = 0


def _corner_rect(points):
    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})
    if len(rows) != 2 or len(cols) != 2:
        return None
    expected = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    return (rows[0], rows[1], cols[0], cols[1]) if set(points) == expected else None


def _find_rect(g: np.ndarray):
    winners = []
    for color in sorted(set(map(int, g.ravel())) - {BG}):
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        if len(pts) == 4:
            rect = _corner_rect(pts)
            if rect is not None:
                winners.append((color, rect))
    assert len(winners) == 1, "expected one rectangular corner color"
    return winners[0]


def _fold_point(r, c, rect):
    r0, r1, c0, c1 = rect
    if r0 <= r <= r1 and c < c0:
        return r, 2 * c0 - c
    if r0 <= r <= r1 and c > c1:
        return r, 2 * c1 - c
    if c0 <= c <= c1 and r < r0:
        return 2 * r0 - r, c
    if c0 <= c <= c1 and r > r1:
        return 2 * r1 - r, c
    corner_r = r0 if r < r0 else r1
    corner_c = c0 if c < c0 else c1
    return 2 * corner_r - r, 2 * corner_c - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique same-color four-corner rectangle, draw its border, then fold
    every external scattered pixel into the rectangle using side-band mirror
    axes or nearest-corner point symmetry.
    """
    g = np.asarray(grid, dtype=int)
    color, rect = _find_rect(g)
    r0, r1, c0, c1 = rect
    out = np.zeros_like(g)
    inside = []
    outside = []
    for r, c in zip(*np.where(g != BG)):
        r, c = int(r), int(c)
        if int(g[r, c]) == color and (r, c) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        if r0 < r < r1 and c0 < c < c1:
            inside.append((r, c, int(g[r, c])))
        else:
            outside.append((r, c, int(g[r, c])))

    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    for r, c, v in inside:
        out[r, c] = v
    for r, c, v in outside:
        nr, nc = _fold_point(r, c, rect)
        if r0 < nr < r1 and c0 < nc < c1:
            out[nr, nc] = v
    return out


def _gen_attempt():
    h, w = random.randint(18, 26), random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    rect_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    r0 = random.randint(5, h - 10)
    r1 = random.randint(r0 + 5, h - 5)
    c0 = random.randint(5, w - 10)
    c1 = random.randint(c0 + 5, w - 5)
    for p in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        grid[p] = rect_color
    used = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    distract_color = random.choice([c for c in range(1, 10) if c != rect_color])
    colors = [c for c in range(1, 10) if c not in (rect_color, distract_color)]

    targets = []
    for _ in range(random.randint(8, 14)):
        zone = random.choice(["L", "R", "T", "B", "TL", "TR", "BL", "BR", "IN"])
        if zone == "IN":
            r, c = random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1)
        elif zone == "L":
            r, c = random.randint(r0 + 1, r1 - 1), random.randint(max(1, 2 * c0 - c1 + 1), c0 - 1)
        elif zone == "R":
            r, c = random.randint(r0 + 1, r1 - 1), random.randint(c1 + 1, min(w - 2, 2 * c1 - c0 - 1))
        elif zone == "T":
            r, c = random.randint(max(1, 2 * r0 - r1 + 1), r0 - 1), random.randint(c0 + 1, c1 - 1)
        elif zone == "B":
            r, c = random.randint(r1 + 1, min(h - 2, 2 * r1 - r0 - 1)), random.randint(c0 + 1, c1 - 1)
        else:
            tr = random.randint(max(1, 2 * r0 - r1 + 1), r0 - 1) if "T" in zone else random.randint(r1 + 1, min(h - 2, 2 * r1 - r0 - 1))
            tc = random.randint(max(1, 2 * c0 - c1 + 1), c0 - 1) if "L" in zone else random.randint(c1 + 1, min(w - 2, 2 * c1 - c0 - 1))
            r, c = tr, tc
        if (r, c) not in used:
            used.add((r, c))
            grid[r, c] = random.choice(colors)

    for p in random.sample([(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if (r, c) not in used], 4):
        grid[p] = distract_color
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
