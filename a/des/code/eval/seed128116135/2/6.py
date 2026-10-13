import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
PALETTE = list(range(10))
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FENCE_COUNT_RANGE = (1, 4)
RECT_H_RANGE = (5, 10)
RECT_W_RANGE = (5, 10)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80
RESERVE_PAD = 1
MAX_SPROUT_LEN = 4


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _border_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _detect_fences(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fences: list[dict] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components_of_color(g, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = set(_border_positions(r0, c0, r1, c1))
            pts = set(comp)
            if not pts <= border:
                continue
            if len(border) - len(pts) != 1:
                continue
            gap = next(iter(border - pts))
            gr, gc = gap
            if gr == r0 and c0 < gc < c1:
                direction = (-1, 0)
            elif gr == r1 and c0 < gc < c1:
                direction = (1, 0)
            elif gc == c0 and r0 < gr < r1:
                direction = (0, -1)
            elif gc == c1 and r0 < gr < r1:
                direction = (0, 1)
            else:
                continue
            fences.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "gap": gap,
                    "direction": direction,
                }
            )
    fences.sort(key=lambda f: (f["bbox"][0], f["bbox"][1], f["color"]))
    return fences


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    for fence in _detect_fences(g):
        color = int(fence["color"])
        r0, c0, r1, c1 = fence["bbox"]
        dr, dc = fence["direction"]

        seeds: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(g[r, c])
                if value != bg and value != color:
                    seeds.append((r, c, value))
        counts = Counter(v for _, _, v in seeds)

        for r, c, seed_color in seeds:
            length = counts[seed_color]
            for step in range(1, length + 1):
                rr = r + dr * step
                cc = c + dc * step
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                    break
                if int(g[rr, cc]) != bg:
                    break
                out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grid contains several rectangular fence loops with exactly one 1-cell gap.
    Seeds inside each fence grow in the direction encoded by the gap side.
    Each seed grows a sprout whose length equals the count of same-colored seeds
    inside that fence, while the sprout color is the fence color.
    """
    return _transform(np.asarray(grid, dtype=int))


def _sprout_cells_for_seed(
    grid: np.ndarray,
    fence_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    seed: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    bg = _get_bg(grid)
    r, c = seed
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for step in range(1, length + 1):
        rr = r + dr * step
        cc = c + dc * step
        if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            break
        if int(grid[rr, cc]) != bg:
            break
        cells.append((rr, cc))
    return cells


def _reserve_box(
    reserve: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    direction: tuple[int, int],
) -> bool:
    dr, dc = direction
    ext_top = MAX_SPROUT_LEN if dr < 0 else 0
    ext_bottom = MAX_SPROUT_LEN if dr > 0 else 0
    ext_left = MAX_SPROUT_LEN if dc < 0 else 0
    ext_right = MAX_SPROUT_LEN if dc > 0 else 0
    rr0 = max(0, r0 - RESERVE_PAD - ext_top)
    rr1 = min(reserve.shape[0] - 1, r1 + RESERVE_PAD + ext_bottom)
    cc0 = max(0, c0 - RESERVE_PAD - ext_left)
    cc1 = min(reserve.shape[1] - 1, c1 + RESERVE_PAD + ext_right)
    if reserve[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
        return False
    reserve[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _make_fence_patch(
    fence_color: int,
    rect_h: int,
    rect_w: int,
    direction: tuple[int, int],
    gap_index: int,
) -> np.ndarray:
    patch = np.full((rect_h, rect_w), BG, dtype=int)
    patch[0, :] = fence_color
    patch[-1, :] = fence_color
    patch[:, 0] = fence_color
    patch[:, -1] = fence_color
    if direction == (-1, 0):
        patch[0, gap_index] = BG
    elif direction == (1, 0):
        patch[-1, gap_index] = BG
    elif direction == (0, -1):
        patch[gap_index, 0] = BG
    else:
        patch[gap_index, -1] = BG
    return patch


def _place_fence(
    rng: random.Random,
    grid: np.ndarray,
    reserve: np.ndarray,
    fence_color: int,
    seed_palette: list[int],
) -> bool:
    h, w = grid.shape
    direction = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    rect_h = rng.randint(*RECT_H_RANGE)
    rect_w = rng.randint(*RECT_W_RANGE)
    rect_h = min(rect_h, h - 2)
    rect_w = min(rect_w, w - 2)
    if rect_h < 5 or rect_w < 5:
        return False

    inner_h = rect_h - 2
    inner_w = rect_w - 2
    lane_capacity = inner_w if direction[0] != 0 else inner_h
    max_seed_count = min(5, inner_h * inner_w, lane_capacity)
    if max_seed_count < 2:
        return False

    total_seeds = rng.randint(2, max_seed_count)
    seed_color_count = rng.randint(1, min(3, total_seeds, len(seed_palette)))
    chosen_seed_colors = rng.sample(seed_palette, seed_color_count)
    seed_colors = chosen_seed_colors[:]
    while len(seed_colors) < total_seeds:
        seed_colors.append(rng.choice(chosen_seed_colors))
    rng.shuffle(seed_colors)

    if direction[0] != 0:
        gap_index = rng.randint(1, rect_w - 2)
        lanes = list(range(1, rect_w - 1))
    else:
        gap_index = rng.randint(1, rect_h - 2)
        lanes = list(range(1, rect_h - 1))

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = rng.randint(1, h - rect_h - 1)
        c0 = rng.randint(1, w - rect_w - 1)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        if not _reserve_box(reserve.copy(), r0, c0, r1, c1, direction):
            continue

        patch = _make_fence_patch(fence_color, rect_h, rect_w, direction, gap_index)

        chosen_lanes = rng.sample(lanes, total_seeds)
        if rng.random() < 0.65 and gap_index in lanes and gap_index not in chosen_lanes:
            chosen_lanes[0] = gap_index
            rng.shuffle(chosen_lanes)

        local_seeds: list[tuple[int, int, int]] = []
        for lane, seed_color in zip(chosen_lanes, seed_colors):
            if direction == (-1, 0):
                lr = rng.randint(1, rect_h - 2)
                lc = lane
            elif direction == (1, 0):
                lr = rng.randint(1, rect_h - 2)
                lc = lane
            elif direction == (0, -1):
                lr = lane
                lc = rng.randint(1, rect_w - 2)
            else:
                lr = lane
                lc = rng.randint(1, rect_w - 2)
            local_seeds.append((lr, lc, seed_color))

        test = np.full_like(grid, BG)
        test[r0 : r1 + 1, c0 : c1 + 1] = patch
        for lr, lc, seed_color in local_seeds:
            test[r0 + lr, c0 + lc] = seed_color

        counts = Counter(seed_color for _, _, seed_color in local_seeds)
        visible_cells = 0
        for lr, lc, seed_color in local_seeds:
            cells = _sprout_cells_for_seed(
                test,
                (r0, c0, r1, c1),
                direction,
                (r0 + lr, c0 + lc),
                counts[seed_color],
            )
            visible_cells += len(cells)
        if visible_cells == 0:
            continue

        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BG):
            continue
        if reserve[r0 : r1 + 1, c0 : c1 + 1].any():
            continue

        if not _reserve_box(reserve, r0, c0, r1, c1, direction):
            continue

        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
        for lr, lc, seed_color in local_seeds:
            grid[r0 + lr, c0 + lc] = seed_color
        return True
    return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserve = np.zeros((h, w), dtype=bool)

    fence_count = rng.randint(*FENCE_COUNT_RANGE)
    colors = [c for c in PALETTE if c != BG]
    rng.shuffle(colors)
    fence_colors = colors[:fence_count]
    seed_palette = colors[fence_count:]
    if len(seed_palette) < 3:
        return None

    for fence_color in fence_colors:
        if not _place_fence(rng, grid, reserve, fence_color, seed_palette):
            return None

    fences = _detect_fences(grid)
    if len(fences) != fence_count:
        return None

    directions = {tuple(fence["direction"]) for fence in fences}
    if not directions:
        return None

    out = _transform(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < fence_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
