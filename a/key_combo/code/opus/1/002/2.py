import random
from collections import Counter, deque

import numpy as np


BG = 0
GRAY = 5
SEED_COLORS = [2, 1, 3]


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


def _frame_boxes(g):
    boxes = []
    for comp in _components(g == GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 >= 4 and c1 - c0 >= 4:
            boxes.append((r0, c0, r1, c1))
    boxes.sort(key=lambda b: (b[0], b[1], -(b[2] - b[0]) * (b[3] - b[1])))
    boxes.sort(key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1), reverse=True)
    return boxes


def _ring_id(r, c, boxes):
    return sum(1 for r0, c0, r1, c1 in boxes if r0 < r < r1 and c0 < c < c1)


def _has_gap(g, box):
    r0, c0, r1, c1 = box
    perim = [(r0, c) for c in range(c0, c1 + 1)]
    perim += [(r1, c) for c in range(c0, c1 + 1)]
    perim += [(r, c0) for r in range(r0 + 1, r1)]
    perim += [(r, c1) for r in range(r0 + 1, r1)]
    return any(g[r, c] != GRAY for r, c in perim)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill only the reachable rings whose geometric ring number has the
    same parity as the seed's ring number; odd/even mismatches are passable but
    not colored.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    boxes = _frame_boxes(g)
    assert boxes, "expected nested gray frames"
    ring_count = len(boxes) + 1

    graph = {i: set() for i in range(ring_count)}
    for i, box in enumerate(boxes):
        if _has_gap(g, box):
            graph[i].add(i + 1)
            graph[i + 1].add(i)

    seeds = []
    for r, c in zip(*np.where((g != BG) & (g != GRAY))):
        r, c = int(r), int(c)
        seeds.append((_ring_id(r, c, boxes), r, c, int(g[r, c])))
    seeds.sort()

    out = g.copy()
    owner: dict[int, int] = {}
    for src_ring, _, _, color in seeds:
        q = deque([src_ring])
        reached = {src_ring}
        while q:
            ring = q.popleft()
            for nb in graph[ring]:
                if nb not in reached:
                    reached.add(nb)
                    q.append(nb)
        for ring in sorted(reached):
            if ring % 2 == src_ring % 2 and ring not in owner:
                owner[ring] = color

    for r in range(h):
        for c in range(w):
            if g[r, c] == BG:
                ring = _ring_id(r, c, boxes)
                if ring in owner:
                    out[r, c] = owner[ring]
    return out


def _draw_frame(grid, box, gap_count):
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    candidates = []
    for c in range(c0 + 2, c1 - 1):
        candidates.append((r0, c))
        candidates.append((r1, c))
    for r in range(r0 + 2, r1 - 1):
        candidates.append((r, c0))
        candidates.append((r, c1))
    for r, c in random.sample(candidates, gap_count):
        grid[r, c] = BG


def _place_seed(grid, boxes, ring, color):
    cells = [
        (r, c)
        for r in range(1, grid.shape[0] - 1)
        for c in range(1, grid.shape[1] - 1)
        if grid[r, c] == BG and _ring_id(r, c, boxes) == ring
    ]
    if not cells:
        return False
    r, c = random.choice(cells)
    grid[r, c] = color
    return True


def _gen_attempt():
    h = w = random.randint(17, 23)
    n = random.randint(3, 4)
    margin = random.randint(1, 2)
    boxes = []
    for i in range(n):
        r0 = margin + i * 2
        c0 = margin + i * 2
        r1 = h - 1 - margin - i * 2
        c1 = w - 1 - margin - i * 2
        if r1 - r0 < 4 or c1 - c0 < 4:
            return None
        boxes.append((r0, c0, r1, c1))

    grid = np.zeros((h, w), dtype=int)
    solid_i = random.randrange(n)
    for i, box in enumerate(boxes):
        if i == solid_i:
            gaps = 0
        else:
            gaps = random.randint(1, 2)
        _draw_frame(grid, box, gaps)

    rings = list(range(n + 1))
    random.shuffle(rings)
    chosen = []
    for ring in rings:
        if ring % 2 not in [x % 2 for x in chosen]:
            chosen.append(ring)
        if len(chosen) >= 2:
            break
    if len(chosen) < 2:
        return None
    if random.random() < 0.65:
        extra = random.choice([r for r in range(n + 1) if r not in chosen])
        chosen.append(extra)
    colors = random.sample(SEED_COLORS, len(chosen))
    for ring, color in zip(chosen, colors):
        if not _place_seed(grid, boxes, ring, color):
            return None

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
