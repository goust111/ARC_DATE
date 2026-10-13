import random
from collections import deque

import numpy as np

BG = 0
GRAY = 5
RED = 2
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _detect_floor_and_side(grid: np.ndarray):
    h, w = grid.shape
    floor_rows = [int(r) for r in np.where(np.all(grid == GRAY, axis=1))[0]]
    if not floor_rows:
        return None, None
    floor_r = floor_rows[0]

    red_coords = np.argwhere(grid == RED)
    if red_coords.size == 0:
        return floor_r, None

    side = None
    if np.any(red_coords[:, 1] == 0):
        side = "left"
    if np.any(red_coords[:, 1] == w - 1):
        side = "right"
    if np.any(red_coords[:, 0] == 0):
        side = "top"

    return floor_r, side


def _mirror_cell(r: int, c: int, h: int, w: int, axis: str):
    if axis == "vert":
        return r, w - 1 - c
    return h - 1 - r, c


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    floor_r, side = _detect_floor_and_side(g)
    if floor_r is None or side is None:
        return g.copy()

    axis = "vert" if side in {"left", "right"} else "hori"

    movable_mask = (g != BG) & (g != GRAY) & (g != RED)
    movable = []
    for comp in _components(movable_mask):
        if not comp:
            continue
        movable.append((list(comp), int(g[comp[0]])))

    mirrored = []
    for comp, color in movable:
        mirrored_cells = []
        for r, c in comp:
            mr, mc = _mirror_cell(r, c, h, w, axis)
            mirrored_cells.append((int(mr), int(mc)))
        mirrored.append((mirrored_cells, color))

    all_objects = movable + mirrored

    out = np.zeros_like(g)
    out[floor_r, :] = GRAY

    blocked = np.zeros((h, w), dtype=bool)
    if 0 <= floor_r < h:
        blocked[floor_r, :] = True
        if floor_r > 0:
            blocked[floor_r - 1, :] = True
    blocked[h - 1, :] = True

    occupied = blocked.copy()
    order = sorted(all_objects, key=lambda t: max(r for r, _ in t[0]), reverse=True)

    for cells, color in order:
        if not cells:
            continue
        arr = np.array(cells, dtype=int)
        shift = 0
        while True:
            candidate = arr + np.array([shift + 1, 0], dtype=int)
            rr = candidate[:, 0]
            cc = candidate[:, 1]
            if np.any(rr >= h) or np.any(cc < 0) or np.any(cc >= w):
                break
            if np.any(occupied[rr, cc]):
                break
            shift += 1

        placed = arr + np.array([shift, 0], dtype=int)
        for r, c in placed:
            out[int(r), int(c)] = color
            occupied[int(r), int(c)] = True

    return out


def _generate_shape(shape_pool):
    return random.choice(shape_pool)


def generate_input() -> np.ndarray:
    h, w = 12, 12
    g = np.zeros((h, w), dtype=int)

    floor_r = random.randint(8, 9)
    g[floor_r, :] = GRAY

    marker_side = random.choice(["left", "right", "top"])
    mark_len = random.randint(2, 3)
    if marker_side == "top":
        c0 = random.randint(0, w - mark_len)
        for dc in range(mark_len):
            g[0, c0 + dc] = RED
    elif marker_side == "left":
        r0 = random.randint(1, h - mark_len - 1)
        for dr in range(mark_len):
            g[r0 + dr, 0] = RED
    else:
        r0 = random.randint(1, h - mark_len - 1)
        for dr in range(mark_len):
            g[r0 + dr, w - 1] = RED

    shapes = [
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ]

    colors = [1, 3, 4, 7]
    random.shuffle(colors)
    target_count = random.randint(2, 3)

    used_cells = set(zip(*np.where((g != BG) & (g != GRAY))))

    def mirror_cells(cells, side):
        if side in {"left", "right"}:
            return [(r, w - 1 - c) for r, c in cells]
        return [(h - 1 - r, c) for r, c in cells]

    placed = 0
    for _ in range(2000):
        if placed >= target_count:
            break

        shape = _generate_shape(shapes)
        color = colors[placed % len(colors)]

        dh = max(r for r, _ in shape)
        dw = max(c for _, c in shape)
        max_top = floor_r - 3 - dh
        if max_top < 1:
            continue

        top = random.randint(1, max_top)
        left = random.randint(0, w - dw - 2)

        cells = [(top + dr, left + dc) for dr, dc in shape]
        mirrored = mirror_cells(cells, marker_side)

        if any((r, c) in used_cells for r, c in cells):
            continue
        if any((r, c) in used_cells for r, c in mirrored):
            continue
        if any(r == floor_r for r, _ in mirrored):
            continue
        if not all(0 <= r < h and 0 <= c < w for r, c in mirrored):
            continue

        for r, c in cells:
            g[r, c] = color
            used_cells.add((r, c))
        for r, c in mirrored:
            used_cells.add((r, c))
        placed += 1

    if placed < 2:
        # deterministic fallback
        fallback_shapes = [
            [(2, 2), (3, 2), (4, 2)],
            [(2, 6), (3, 6)],
        ]
        for i, shape in enumerate(fallback_shapes):
            if i >= len(colors):
                break
            color = colors[i]
            for r, c in shape:
                g[r, c] = color

    return g
