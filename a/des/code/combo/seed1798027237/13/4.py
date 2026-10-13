from collections import deque
import random

import numpy as np


BG = 8
COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
NBS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _bbox(coords):
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NBS8:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append({"color": color, "mask": mask})
    return comps


def _stack_rectangles(rects):
    rects = sorted(rects, key=lambda item: (item[0], item[1], item[2]), reverse=True)
    out_h = max(h for _, h, _, _ in rects)
    out_w = max(w for _, _, w, _ in rects)
    out = np.full((out_h, out_w), BG, dtype=int)
    for _, h, w, patch in rects:
        r0 = (out_h - h) // 2
        c0 = (out_w - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid)
    unused = set(range(len(comps)))
    rects = []

    while unused:
        i = min(unused)
        unused.remove(i)
        a = comps[i]
        mate = None
        for j in sorted(unused):
            b = comps[j]
            if a["color"] == b["color"] or a["mask"].shape != b["mask"].shape:
                continue
            if np.array_equal(a["mask"], ~b["mask"]):
                mate = j
                break
        if mate is None:
            continue

        unused.remove(mate)
        b = comps[mate]
        h, w = a["mask"].shape
        patch = np.empty((h, w), dtype=int)
        patch[a["mask"]] = a["color"]
        patch[b["mask"]] = b["color"]
        rects.append((h * w, h, w, patch))

    if not rects:
        return np.zeros((0, 0), dtype=int)
    return _stack_rectangles(rects)


def _connected(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in NBS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(pts)


def _touches_all_sides(mask):
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_complement_mask(h, w):
    for _ in range(5000):
        mask = np.random.random((h, w)) < random.uniform(0.42, 0.58)
        area = int(mask.sum())
        if area < 4 or h * w - area < 4:
            continue
        inv = ~mask
        if (
            _touches_all_sides(mask)
            and _touches_all_sides(inv)
            and _connected(mask)
            and _connected(inv)
        ):
            return mask
    raise RuntimeError("failed to build complementary masks")


def _place_patch(canvas, mask, color):
    h, w = mask.shape
    rows, cols = canvas.shape
    for _ in range(1000):
        r0 = random.randrange(1, rows - h - 1)
        c0 = random.randrange(1, cols - w - 1)
        padded = canvas[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1]
        if np.all(padded == BG):
            canvas[r0 : r0 + h, c0 : c0 + w][mask] = color
            return True
    return False


def _gen_attempt():
    pair_count = random.randint(3, 5)
    dims = []
    used_shapes = set()
    used_areas = set()
    for _ in range(pair_count):
        for _ in range(100):
            h = random.randint(4, 8)
            w = random.randint(4, 9)
            if (h, w) not in used_shapes and h * w not in used_areas:
                used_shapes.add((h, w))
                used_areas.add(h * w)
                dims.append((h, w))
                break
        else:
            return None

    canvas = np.full((30, 30), BG, dtype=int)
    rects = []
    colors = COLORS[:]
    random.shuffle(colors)

    for idx, (h, w) in enumerate(dims):
        mask = _random_complement_mask(h, w)
        c1 = colors[(2 * idx) % len(colors)]
        c2 = colors[(2 * idx + 1) % len(colors)]
        if c1 == c2:
            c2 = COLORS[(COLORS.index(c1) + 1) % len(COLORS)]
        if not _place_patch(canvas, mask, c1):
            return None
        if not _place_patch(canvas, ~mask, c2):
            return None

        patch = np.empty((h, w), dtype=int)
        patch[mask] = c1
        patch[~mask] = c2
        rects.append((h * w, h, w, patch))

    expected = _stack_rectangles(rects)
    if not np.array_equal(main(canvas), expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid complementary-fragment input")
