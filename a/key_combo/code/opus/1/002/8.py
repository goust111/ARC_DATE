import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED_FRAME = 2
GREEN_FRAME = 3
ORANGE = 7
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _frames(g):
    frames = []
    for color in (RED_FRAME, GREEN_FRAME):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 == 7 and c1 - c0 + 1 == 7:
                frames.append({"color": color, "box": (r0, c0, r1, c1), "inside": g[r0 + 1 : r1, c0 + 1 : c1].copy()})
    assert len(frames) == 3, "expected three 7x7 frames"
    return frames


def _infer_pair(pair):
    a, b = pair
    a_inside = a["inside"]
    b_inside = b["inside"]
    assert np.array_equal(a_inside == BLUE, b_inside == BLUE), "training sprites must match"
    a_orange = a_inside == ORANGE
    b_orange = b_inside == ORANGE
    if np.all(a_orange <= b_orange) and int(a_orange.sum()) < int(b_orange.sum()):
        return a_inside, b_inside
    if np.all(b_orange <= a_orange) and int(b_orange.sum()) < int(a_orange.sum()):
        return b_inside, a_inside
    raise AssertionError("could not order training pair")


def _infer_direction(before, after):
    markers = [(int(r), int(c)) for r, c in zip(*np.where(before == ORANGE))]
    extra = {(int(r), int(c)) for r, c in zip(*np.where((after == ORANGE) & (before != ORANGE)))}
    scores = defaultdict(int)
    for mr, mc in markers:
        for er, ec in extra:
            if er == mr and ec != mc:
                scores[(0, 1 if ec > mc else -1)] += 1
            if ec == mc and er != mr:
                scores[(1 if er > mr else -1, 0)] += 1
    assert scores, "missing ray evidence"
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _apply_ray(interior, direction):
    out = interior.copy()
    markers = [(int(r), int(c)) for r, c in zip(*np.where(interior == ORANGE))]
    dr, dc = direction
    for mr, mc in markers:
        r, c = mr + dr, mc + dc
        while 0 <= r < 5 and 0 <= c < 5:
            if out[r, c] == BLUE:
                break
            out[r, c] = ORANGE
            r += dr
            c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the same-color frame pair, infer its orange ray direction by comparing
    before and after interiors, and apply that ray rule to the unique test
    frame. Return only the transformed 5x5 test interior.
    """
    g = np.asarray(grid, dtype=int)
    frames = _frames(g)
    by_color = defaultdict(list)
    for frame in frames:
        by_color[frame["color"]].append(frame)
    train = next(v for v in by_color.values() if len(v) == 2)
    test = next(v[0] for v in by_color.values() if len(v) == 1)
    before, after = _infer_pair(train)
    direction = _infer_direction(before, after)
    return _apply_ray(test["inside"], direction)


def _draw_frame(grid, top, left, color):
    grid[top : top + 7, left] = color
    grid[top : top + 7, left + 6] = color
    grid[top, left : left + 7] = color
    grid[top + 6, left : left + 7] = color


def _make_interior(direction, markers=1, variant=0):
    dr, dc = direction
    arr = np.zeros((5, 5), dtype=int)
    if direction == (0, 1):
        blue = [(1, 1), (2, 1), (3, 1), (1, 2)]
        marks = [(2, 2), (3, 2)]
    elif direction == (0, -1):
        blue = [(1, 3), (2, 3), (3, 3), (1, 2)]
        marks = [(2, 2), (3, 2)]
    elif direction == (1, 0):
        blue = [(1, 1), (1, 2), (1, 3), (2, 1)]
        marks = [(2, 2), (2, 3)]
    else:
        blue = [(3, 1), (3, 2), (3, 3), (2, 1)]
        marks = [(2, 2), (2, 3)]
    if variant:
        blue = [(r, 4 - c) if variant == 1 else (4 - r, c) for r, c in blue]
    for r, c in blue:
        arr[r, c] = BLUE
    for r, c in marks[:markers]:
        arr[r, c] = ORANGE
    return arr


def _place_interior(grid, top, left, interior):
    grid[top + 1 : top + 6, left + 1 : left + 6] = interior


def _positions():
    choices = [
        [(1, 1), (1, 12), (12, 1)],
        [(1, 2), (12, 2), (7, 13)],
        [(2, 1), (2, 13), (13, 7)],
        [(1, 1), (11, 11), (13, 1)],
    ]
    return random.choice(choices)


def _gen_attempt():
    grid = np.zeros((21, 21), dtype=int)
    pos = _positions()
    direction = random.choice(DIRS)
    marker_count = random.randint(1, 2)
    before = _make_interior(direction, marker_count, 0)
    after = _apply_ray(before, direction)
    test = _make_interior(direction, marker_count, random.choice([1, 2]))
    random.shuffle(pos)
    specs = [
        (pos[0], RED_FRAME, before),
        (pos[1], RED_FRAME, after),
        (pos[2], GREEN_FRAME, test),
    ]
    for (top, left), color, interior in specs:
        _draw_frame(grid, top, left, color)
        _place_interior(grid, top, left, interior)
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
