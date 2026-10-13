import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _find_containers(grid: np.ndarray):
    comps = _components(grid == FRAME)
    containers = []
    for cells in comps:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 != 2:
            continue
        if np.all(grid[r1, c0 : c1 + 1] == FRAME) and np.all(grid[r0 : r1 + 1, c0] == FRAME) and np.all(
            grid[r0 : r1 + 1, c1] == FRAME
        ):
            label_r = r1 + 1
            label_c = c0 + 1
            if label_r < grid.shape[0]:
                label = int(grid[label_r, label_c])
                if label not in (BG, FRAME):
                    containers.append((c0, {"bbox": (r0, c0, r1, c1), "label": label}))
    containers.sort(key=lambda item: item[0])
    return [info for _, info in containers]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the stacked colored balls into the label-matched containers while preserving source order.

    Input structure:
    - The background is black and several grey U-shaped containers stand along the bottom of the grid.
    - Each container has one interior column where colored single-pixel balls are stacked from bottom to top.
    - A single colored label pixel sits just below each container and indicates that container's target color.

    Rule:
    - Move every ball into the unique container whose label color matches the ball color.
    - Within one source container, balls of the same destination color keep their original bottom-to-top order.
    - Across different source containers, groups are appended in source left-to-right order, while the grey container walls and label pixels stay fixed.

    Key insight:
    - Color decides the destination container, but the final stack order is governed by source-container order, not by the colors themselves.
    """
    g = np.asarray(grid, dtype=int)
    containers = _find_containers(g)
    by_label = {item["label"]: [] for item in containers}
    centers = {}
    out = g.copy()

    for item in containers:
        r0, c0, r1, c1 = item["bbox"]
        center_c = (c0 + c1) // 2
        centers[item["label"]] = (r1, center_c)
        interior_rows = range(r1 - 1, r0, -1)
        for rr in interior_rows:
            color = int(g[rr, center_c])
            if color not in (BG, FRAME):
                by_label[color].append(color)
                out[rr, center_c] = BG

    for item in containers:
        label = item["label"]
        r1, center_c = centers[label]
        for idx, color in enumerate(by_label[label]):
            out[r1 - 1 - idx, center_c] = color
    return out


def _gen_attempt(rng: random.Random):
    count = rng.randint(3, 5)
    colors = rng.sample(PALETTE, count)
    inner_h = rng.randint(3, 6)
    container_h = inner_h + 1
    width = count * 3 + (count - 1) * rng.randint(1, 2)
    height = container_h + 3
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    gap = (width - count * 3) // max(1, count - 1) if count > 1 else 0
    xs = []
    cur = 0
    for _ in range(count):
        xs.append(cur)
        cur += 3 + gap

    total_capacity = inner_h * count
    if total_capacity < count:
        return None

    ball_lists = []
    all_balls = []
    for _ in range(count):
        size = rng.randint(1, inner_h)
        balls = [rng.choice(colors) for _ in range(size)]
        ball_lists.append(balls)
        all_balls.extend(balls)
    for color in colors:
        if color not in all_balls:
            target = rng.randrange(count)
            slot = rng.randrange(len(ball_lists[target]))
            ball_lists[target][slot] = color

    top = 0
    bottom = container_h
    for idx, x in enumerate(xs):
        grid[top:bottom + 1, x] = FRAME
        grid[top:bottom + 1, x + 2] = FRAME
        grid[bottom, x : x + 3] = FRAME
        grid[bottom + 1, x + 1] = colors[idx]
        for offset, color in enumerate(ball_lists[idx]):
            grid[bottom - 1 - offset, x + 1] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
