import numpy as np
from collections import deque


BG = 0
TRAIL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _components_by_color(grid: np.ndarray):
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
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, pts))
    return comps


def _bbox(points):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _opening_direction(points):
    r0, c0, r1, c1 = _bbox(points)
    point_set = set(points)
    counts = {
        "up": sum((r0, c) in point_set for c in range(c0, c1 + 1)),
        "down": sum((r1, c) in point_set for c in range(c0, c1 + 1)),
        "left": sum((r, c0) in point_set for r in range(r0, r1 + 1)),
        "right": sum((r, c1) in point_set for r in range(r0, r1 + 1)),
    }
    min_count = min(counts.values())
    missing = [name for name, count in counts.items() if count == min_count]
    assert len(missing) == 1, f"expected exactly one open side, got {counts}"
    return missing[0]


def _shift_to_edge(points, direction, shape):
    h, w = shape
    r0, c0, r1, c1 = _bbox(points)
    if direction == "up":
        return -r0, 0
    if direction == "down":
        return h - 1 - r1, 0
    if direction == "left":
        return 0, -c0
    if direction == "right":
        return 0, w - 1 - c1
    raise ValueError(direction)


def _trail_color(step_idx: int, total_steps: int, obj_color: int) -> int:
    if total_steps <= 1:
        return obj_color
    return obj_color if (2 * step_idx >= total_steps) else TRAIL


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BG)

    for color, pts in _components_by_color(g):
        direction = _opening_direction(pts)
        dy, dx = _shift_to_edge(pts, direction, g.shape)
        steps = abs(dy) + abs(dx)

        for step in range(1, steps):
            rr_off = 0 if dy == 0 else (1 if dy > 0 else -1) * step
            cc_off = 0 if dx == 0 else (1 if dx > 0 else -1) * step
            trail_color = _trail_color(step, steps, color)
            for r, c in pts:
                rr, cc = r + rr_off, c + cc_off
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = trail_color

        for r, c in pts:
            rr, cc = r + dy, c + dx
            out[rr, cc] = color

    return out


def _make_shape(direction: str, height: int, width: int) -> np.ndarray:
    shape = np.zeros((height, width), dtype=int)
    if direction == "up":
        shape[1:, 0] = 1
        shape[1:, -1] = 1
        shape[-1, :] = 1
    elif direction == "down":
        shape[:-1, 0] = 1
        shape[:-1, -1] = 1
        shape[0, :] = 1
    elif direction == "left":
        shape[0, 1:] = 1
        shape[-1, 1:] = 1
        shape[:, -1] = 1
    elif direction == "right":
        shape[0, :-1] = 1
        shape[-1, :-1] = 1
        shape[:, 0] = 1
    else:
        raise ValueError(direction)
    return shape


def _swept_cells(top: int, left: int, shape: np.ndarray, direction: str, steps: int):
    cells = set()
    pts = list(map(tuple, np.argwhere(shape == 1)))
    for step in range(steps + 1):
        if direction == "up":
            dr, dc = -step, 0
        elif direction == "down":
            dr, dc = step, 0
        elif direction == "left":
            dr, dc = 0, -step
        else:
            dr, dc = 0, step
        for r, c in pts:
            cells.add((top + r + dr, left + c + dc))
    return cells


def _random_shape(rng: np.random.Generator, direction: str) -> np.ndarray:
    if direction in ("up", "down"):
        height = int(rng.integers(4, 7))
        width = int(rng.integers(3, 7))
    else:
        height = int(rng.integers(3, 7))
        width = int(rng.integers(4, 7))
    return _make_shape(direction, height, width)


def _place_object(grid, occupied, rng, direction, color):
    h, w = grid.shape
    shape = _random_shape(rng, direction)
    sh, sw = shape.shape

    if direction == "up":
        step_min = 2
        step_max = h - sh - 1
        if step_max < step_min:
            return False
        top = int(rng.integers(step_min, step_max + 1))
        left = int(rng.integers(1, w - sw))
        steps = top
    elif direction == "down":
        step_min = 2
        step_max = h - sh - 1
        if step_max < step_min:
            return False
        top = int(rng.integers(0, h - sh - step_min + 1))
        left = int(rng.integers(1, w - sw))
        steps = h - (top + sh)
    elif direction == "left":
        step_min = 2
        step_max = w - sw - 1
        if step_max < step_min:
            return False
        top = int(rng.integers(1, h - sh))
        left = int(rng.integers(step_min, step_max + 1))
        steps = left
    else:
        step_min = 2
        step_max = w - sw - 1
        if step_max < step_min:
            return False
        top = int(rng.integers(1, h - sh))
        left = int(rng.integers(0, w - sw - step_min + 1))
        steps = w - (left + sw)

    swept = _swept_cells(top, left, shape, direction, steps)
    if any((r, c) in occupied for r, c in swept):
        return False

    for rr, cc in swept:
        occupied.add((rr, cc))
    for r, c in np.argwhere(shape == 1):
        grid[top + r, left + c] = color
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        h = int(rng.integers(12, 19))
        w = int(rng.integers(12, 19))
        grid = np.zeros((h, w), dtype=int)
        occupied = set()

        obj_count = int(rng.integers(2, 5))
        directions = list(rng.choice(["up", "down", "left", "right"], size=obj_count, replace=False))
        colors = list(rng.choice(PALETTE, size=obj_count, replace=False))

        ok = True
        for direction, color in zip(directions, colors):
            placed = False
            for _try in range(60):
                if _place_object(grid, occupied, rng, direction, int(color)):
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        comps = _components_by_color(grid)
        if len(comps) != obj_count:
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate input")
