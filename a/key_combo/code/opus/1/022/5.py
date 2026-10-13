import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _frame_cells(box):
    r0, r1, c0, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _toposort(colors, above):
    remaining = set(colors)
    order = []
    while remaining:
        bottoms = [c for c in remaining if not any((x, c) in above for x in remaining)]
        if not bottoms:
            order.extend(sorted(remaining))
            break
        c = sorted(bottoms)[0]
        order.append(c)
        remaining.remove(c)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    template_start = max(0, h - 3)
    frame_zone = grid[:template_start, :]
    colors = [int(c) for c in np.unique(frame_zone) if c not in (0,)]
    boxes = {}
    frames = {}
    for color in colors:
        pts = np.argwhere(frame_zone == color)
        if len(pts) == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        boxes[color] = (int(r0), int(r1), int(c0), int(c1))
        frames[color] = _frame_cells(boxes[color])
    above = set()
    for r in range(template_start):
        for c in range(w):
            covered = [color for color, cells in frames.items() if (r, c) in cells]
            if len(covered) > 1 and grid[r, c] in covered:
                visible = int(grid[r, c])
                for color in covered:
                    if color != visible:
                        above.add((color, visible))
    order = _toposort(list(frames), above)
    mapped = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))} if order else {}

    templates = {}
    zone = grid[template_start:, :]
    for color in frames:
        comps = _components(zone == color)
        if comps:
            comp = max(comps, key=len)
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            templates[color] = [(r - min(rs), c - min(cs)) for r, c in comp]

    out = grid.copy()
    for r0, r1, c0, c1 in boxes.values():
        out[r0 + 1:r1, c0 + 1:c1] = 0
    out[template_start:, :] = 0
    for color, box in boxes.items():
        if color not in templates:
            continue
        pts = templates[color]
        th = max(r for r, _ in pts) + 1
        tw = max(c for _, c in pts) + 1
        r0, r1, c0, c1 = box
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        ar = r0 + 1 + (ih - th) // 2
        ac = c0 + 1 + (iw - tw) // 2
        for dr, dc in pts:
            rr, cc = ar + dr, ac + dc
            if out[rr, cc] == 0:
                out[rr, cc] = mapped.get(color, color)
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(17, 20)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4], random.randint(3, 4))
    size = random.randint(6, 8)
    starts = [(1, 1), (3, 4), (5, 2), (2, 6)]
    random.shuffle(starts)
    order = colors[:]
    random.shuffle(order)
    boxes = {}
    for color, (r0, c0) in zip(order, starts):
        boxes[color] = (r0, r0 + size - 1, c0, c0 + size - 1)
    for color in order:
        for r, c in _frame_cells(boxes[color]):
            if r < h - 4 and c < w:
                grid[r, c] = color
    x = 1
    for color in colors:
        pts = [(0, 0), (1, 0), (1, 1)]
        if random.random() < 0.5:
            pts.append((2, 1))
        if random.random() < 0.5:
            pts.append((0, 1))
        for dr, dc in pts[:random.randint(3, min(5, len(pts)))]:
            grid[h - 3 + dr, x + dc] = color
        x += 4
    return grid
