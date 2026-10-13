import random
from collections import deque

import numpy as np


BG = 7
BLUE = 1
RED = 2
GREEN = 3
COUNT_TO_COLOR = {2: 4, 3: 5, 4: 2, 5: 6}
NEXT = {"TL": "TR", "TR": "BR", "BR": "BL", "BL": "TL"}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _corner(bbox, h, w):
    r0, c0, r1, c1 = bbox
    if r0 == 0 and c0 == 0:
        return "TL"
    if r0 == 0 and c1 == w - 1:
        return "TR"
    if r1 == h - 1 and c1 == w - 1:
        return "BR"
    if r1 == h - 1 and c0 == 0:
        return "BL"
    return None


def _anchor(corner, h, w, ph, pw):
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BR":
        return h - ph, w - pw
    return h - ph, 0


def _green_count(sub: np.ndarray) -> int:
    return len(_components(sub == GREEN))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy corner objects clockwise, then recolor blue pixels by the post-copy green marker count.

    Input structure:
    - The background is orange, with one blue-and-green object anchored in each true corner.
    - Green pixels are isolated marker components embedded in or adjacent to the blue object body.
    - Red interior objects are distractors and do not participate.

    Rule:
    - Copy every original corner object clockwise to the next corner, preserving orientation and aligning bounding boxes to that corner.
    - After all copies are stamped, count green 4-connected marker components in each quadrant.
    - Recolor all blue pixels in a quadrant according to marker count: 2->yellow, 3->grey, 4->red, 5->magenta.
    - Preserve green markers, orange background, and red distractors.

    Key insight:
    - The green count is measured after copying, so the copy step changes the recoloring key for each corner.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    sources = {}
    for comp in _components((g == BLUE) | (g == GREEN)):
        bbox = _bbox(comp)
        corner = _corner(bbox, h, w)
        if corner is not None:
            sources[corner] = (comp, bbox)
    assert sources, "expected corner sources"

    for corner, (comp, bbox) in sources.items():
        r0, c0, r1, c1 = bbox
        ph, pw = r1 - r0 + 1, c1 - c0 + 1
        dr0, dc0 = _anchor(NEXT[corner], h, w, ph, pw)
        for r, c in comp:
            out[dr0 + r - r0, dc0 + c - c0] = g[r, c]

    mid_r, mid_c = h // 2, w // 2
    qs = {
        "TL": (slice(0, mid_r), slice(0, mid_c)),
        "TR": (slice(0, mid_r), slice(mid_c, w)),
        "BL": (slice(mid_r, h), slice(0, mid_c)),
        "BR": (slice(mid_r, h), slice(mid_c, w)),
    }
    for sl in qs.values():
        sub = out[sl]
        count = _green_count(sub)
        assert count in COUNT_TO_COLOR, f"unsupported marker count {count}"
        sub[sub == BLUE] = COUNT_TO_COLOR[count]
    return out


BODY = {
    (0, 2),
    (0, 4),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 2),
    (3, 2),
    (4, 2),
}
MARKS = {
    "A": (0, 1),
    "B": (0, 3),
    "C": (2, 0),
    "D": (2, 4),
    "E": (4, 1),
}
SETS = {
    "TL": ["A"],
    "TR": ["B", "C"],
    "BR": ["A", "D", "E"],
    "BL": ["B"],
}


def _stamp_object(grid, corner, marks):
    ph = pw = 5
    r0, c0 = _anchor(corner, grid.shape[0], grid.shape[1], ph, pw)
    for rr, cc in BODY:
        grid[r0 + rr, c0 + cc] = BLUE
    for name in marks:
        rr, cc = MARKS[name]
        grid[r0 + rr, c0 + cc] = GREEN


def _gen_attempt():
    n = random.randint(14, 18)
    grid = np.full((n, n), BG, dtype=int)
    for corner, marks in SETS.items():
        _stamp_object(grid, corner, marks)
    for _ in range(random.randint(1, 2)):
        r = random.randint(6, n - 7)
        c = random.randint(6, n - 7)
        grid[r : r + 2, c : c + 2] = RED
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(100):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
