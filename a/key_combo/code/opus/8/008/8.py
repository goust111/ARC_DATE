import random
from collections import Counter, deque

import numpy as np


BG = 0
OBSTACLE = 5
TRIGGER = 6
RAY = 7
FILL = 8
LINE_COLORS = (1, 2, 3, 4, 9)
MAX_ATTEMPTS = 400


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    assert abs(r1 - r0) == abs(c1 - c0) and (r0, c0) != (r1, c1)
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _background_fill_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for start in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        r, c = start
        if grid[r, c] == BG and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == BG:
                seen[nr, nc] = True
                q.append((nr, nc))
    return (grid == BG) & (~seen)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal endpoint pairs, split magenta triggers into perpendicular diagonal rays, and fill enclosed background pockets.

    Input structure:
    - The grid has black background, 2-3 same-color endpoint pairs on 45-degree diagonals, 1-3 magenta trigger pixels, and optional gray obstacle blocks.
    - Each non-special pair color appears as two isolated endpoints, and each trigger lies on exactly one would-be diagonal connection.
    - Obstacles are solid non-background blockers that may help complete a closed boundary.

    Rule:
    - For every ordinary color pair, draw the full diagonal segment between its two endpoints, leaving any trigger pixel on that segment magenta.
    - For each trigger, infer the hosting segment orientation and emit orange rays along the perpendicular diagonal in both directions until a non-background cell or the grid boundary stops the ray.
    - After all segments and rays are drawn, flood-fill background from the four corners and recolor every remaining unvisited background cell teal.

    Key insight:
    - The teal output is not based on a named object; it is the background trapped only after diagonal segments, split rays, and blockers are overlaid into one combined enclosure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    specials = {BG, OBSTACLE, TRIGGER}
    color_points: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(~np.isin(g, list(specials))):
        color_points.setdefault(int(g[r, c]), []).append((int(r), int(c)))

    segments: list[tuple[int, list[tuple[int, int]], tuple[int, int]]] = []
    trigger_dirs: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for color, pts in color_points.items():
        assert len(pts) >= 2, "each line color needs a diagonal pair"
        used = False
        if len(pts) == 2:
            candidates = [tuple(pts)]
        else:
            candidates = []
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    candidates.append((pts[i], pts[j]))
        for p, q in candidates:
            if abs(p[0] - q[0]) != abs(p[1] - q[1]) or p == q:
                continue
            path = _diag_path(p, q)
            dr = 1 if q[0] > p[0] else -1
            dc = 1 if q[1] > p[1] else -1
            segments.append((color, path, (dr, dc)))
            used = True
            break
        assert used, f"color {color} must form a diagonal pair"

    for color, path, (dr, dc) in segments:
        perp_dirs = {(dr, -dc), (-dr, dc)}
        for r, c in path:
            if out[r, c] == BG:
                out[r, c] = color
            elif out[r, c] == TRIGGER:
                trigger_dirs.setdefault((r, c), set()).update(perp_dirs)

    for r, c in np.argwhere(g == TRIGGER):
        key = (int(r), int(c))
        assert key in trigger_dirs, "every trigger must lie on a connected diagonal"
        for dr, dc in sorted(trigger_dirs[key]):
            rr, cc = key[0] + dr, key[1] + dc
            while 0 <= rr < h and 0 <= cc < w and out[rr, cc] == BG:
                out[rr, cc] = RAY
                rr += dr
                cc += dc

    out[_background_fill_mask(out)] = FILL
    return out


def _apply_transform(
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
    mode: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    h, w = shape

    def tf(r: int, c: int) -> tuple[int, int]:
        if mode == 0:
            return r, c
        if mode == 1:
            return c, h - 1 - r
        if mode == 2:
            return h - 1 - r, w - 1 - c
        if mode == 3:
            return w - 1 - c, r
        if mode == 4:
            return r, w - 1 - c
        if mode == 5:
            return h - 1 - r, c
        if mode == 6:
            return c, r
        return w - 1 - c, h - 1 - r

    transformed = [tf(r, c) for r, c in cells]
    max_r = max(r for r, _ in transformed)
    max_c = max(c for _, c in transformed)
    return transformed, (max_r + 1, max_c + 1)


def _paint_obstacle(grid: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    cells = []
    hh, ww = shape
    for r in range(top, top + hh):
        for c in range(left, left + ww):
            grid[r, c] = OBSTACLE
            cells.append((r, c))
    return cells


def _base_scene() -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[int, int]], list[int]]:
    local = np.full((16, 16), BG, dtype=int)
    colors = random.sample(LINE_COLORS, 3)

    pairs = [
        (colors[0], (2, 5), (10, 13)),
        (colors[1], (5, 2), (13, 10)),
    ]
    triggers = [(6, 9)]
    occupied = set(triggers)
    endpoints = []
    for color, p, q in pairs:
        local[p] = color
        local[q] = color
        endpoints.extend([p, q])
        occupied.update([p, q])

    for t in triggers:
        local[t] = TRIGGER

    obstacle_cells = _paint_obstacle(local, 9, 7, random.choice(((1, 3), (2, 2))))
    occupied.update(obstacle_cells)

    if random.random() < 0.55:
        extra_pair = (colors[2], (2, 1), (5, 4))
        pairs.append(extra_pair)
        local[extra_pair[1]] = colors[2]
        local[extra_pair[2]] = colors[2]
        occupied.update([extra_pair[1], extra_pair[2]])
        endpoints.extend([extra_pair[1], extra_pair[2]])
        if random.random() < 0.6:
            local[extra_pair[1][0] + 1, extra_pair[1][1] + 1] = TRIGGER
            triggers.append((extra_pair[1][0] + 1, extra_pair[1][1] + 1))
            occupied.add(triggers[-1])
    else:
        if random.random() < 0.45:
            local[9, 12] = TRIGGER
            triggers.append((9, 12))
            occupied.add((9, 12))

    if len(triggers) < 3 and random.random() < 0.45:
        for candidate in ((11, 8), (4, 7), (8, 11)):
            if candidate not in occupied:
                local[candidate] = TRIGGER
                triggers.append(candidate)
                occupied.add(candidate)
                break

    if random.random() < 0.5:
        cells = _paint_obstacle(local, 2, 11, random.choice(((2, 2), (3, 1), (1, 3))))
        occupied.update(cells)

    return local, triggers, occupied, colors


def _place_distractor_pair(grid: np.ndarray, used: set[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for _ in range(120):
        length = random.randint(3, 5)
        dr, dc = random.choice(((1, 1), (1, -1)))
        r0 = random.randint(0, h - 1)
        c0 = random.randint(0, w - 1)
        r1 = r0 + dr * length
        c1 = c0 + dc * length
        if not (0 <= r1 < h and 0 <= c1 < w):
            continue
        path = _diag_path((r0, c0), (r1, c1))
        if any(cell in used for cell in path):
            continue
        trigger = None
        if length >= 4 and random.random() < 0.7:
            trigger = path[random.randint(1, length - 1)]
        if trigger is not None:
            perp = {(dr, -dc), (-dr, dc)}
            ray_cells = []
            ok = True
            for rr, cc in perp:
                r, c = trigger[0] + rr, trigger[1] + cc
                while 0 <= r < h and 0 <= c < w:
                    if (r, c) in used or (r, c) in path:
                        ok = False
                        break
                    ray_cells.append((r, c))
                    r += rr
                    c += cc
                if not ok:
                    break
            if not ok:
                continue
        grid[r0, c0] = color
        grid[r1, c1] = color
        used.update({(r0, c0), (r1, c1)})
        used.update(path)
        if trigger is not None:
            grid[trigger] = TRIGGER
            used.add(trigger)
            used.update(ray_cells)
        return True
    return False


def _generate_attempt() -> np.ndarray | None:
    base, _triggers, _occupied, colors = _base_scene()
    mode = random.randrange(8)
    nz = [(int(r), int(c)) for r, c in np.argwhere(base != BG)]
    transformed, (th, tw) = _apply_transform(nz, base.shape, mode)
    values = [int(base[r, c]) for r, c in nz]

    margin_h = random.randint(0, 6)
    margin_w = random.randint(0, 6)
    h = min(30, th + 2 + margin_h)
    w = min(30, tw + 2 + margin_w)
    if h < th + 1 or w < tw + 1:
        return None

    top = random.randint(0, h - th)
    left = random.randint(0, w - tw)
    grid = np.full((h, w), BG, dtype=int)
    used: set[tuple[int, int]] = set()
    for (r, c), value in zip(transformed, values):
        rr, cc = top + r, left + c
        if grid[rr, cc] != BG:
            return None
        grid[rr, cc] = value
        used.add((rr, cc))

    unused_colors = [c for c in LINE_COLORS if c not in Counter(values)]
    if unused_colors and random.random() < 0.6:
        _place_distractor_pair(grid, used, random.choice(unused_colors))

    try:
        out = main(grid)
    except AssertionError:
        return None

    fill_count = int(np.sum(out == FILL))
    ray_count = int(np.sum(out == RAY))
    if fill_count < 2 or ray_count < 2:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
