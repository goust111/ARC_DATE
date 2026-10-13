import random
from collections import Counter

import numpy as np


BG = 0
YELLOW = 4
CONTAINER_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
OPENINGS = ("up", "down", "left", "right")
PATTERNS = ("continuous", "alternating")

GRID_H_RANGE = (14, 26)
GRID_W_RANGE = (14, 26)
OUTER_H_RANGE = (5, 9)
OUTER_W_RANGE = (5, 9)
CORRIDOR_DEPTH_RANGE = (2, 6)
CONTAINER_COUNT_RANGE = (1, 4)
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _border_path(height: int, width: int, opening: str) -> list[tuple[int, int]]:
    if opening == "up":
        left = [(r, 0) for r in range(height)]
        bottom = [(height - 1, c) for c in range(1, width)]
        right = [(r, width - 1) for r in range(height - 2, -1, -1)]
        return left + bottom + right
    if opening == "down":
        left = [(r, 0) for r in range(height - 1, -1, -1)]
        top = [(0, c) for c in range(1, width)]
        right = [(r, width - 1) for r in range(1, height)]
        return left + top + right
    if opening == "left":
        top = [(0, c) for c in range(width)]
        right = [(r, width - 1) for r in range(1, height)]
        bottom = [(height - 1, c) for c in range(width - 2, -1, -1)]
        return top + right + bottom
    if opening == "right":
        top = [(0, c) for c in range(width - 1, -1, -1)]
        left = [(r, 0) for r in range(1, height)]
        bottom = [(height - 1, c) for c in range(1, width)]
        return top + left + bottom
    raise ValueError(opening)


def _expected_border_positions(r0: int, c0: int, r1: int, c1: int, opening: str) -> set[tuple[int, int]]:
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    return {(r0 + dr, c0 + dc) for dr, dc in _border_path(height, width, opening)}


def _container_mask(height: int, width: int, opening: str, pattern: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    path = _border_path(height, width, opening)
    if pattern == "continuous":
        chosen = path
    else:
        chosen = path[::2]
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _opening_from_bbox(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> str | None:
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    side_counts = {
        "up": int(np.sum(grid[r0, c0 + 1:c1] == color)),
        "down": int(np.sum(grid[r1, c0 + 1:c1] == color)),
        "left": int(np.sum(grid[r0 + 1:r1, c0] == color)),
        "right": int(np.sum(grid[r0 + 1:r1, c1] == color)),
    }
    openings = [side for side, count in side_counts.items() if count == 0]
    if len(openings) != 1:
        return None
    opening = openings[0]
    if any(side_counts[side] == 0 for side in OPENINGS if side != opening):
        return None
    return opening


def _detect_pattern(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int, opening: str) -> str | None:
    expected = _expected_border_positions(r0, c0, r1, c1, opening)
    pts = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
    if not pts or not pts.issubset(expected):
        return None
    if all(int(grid[r, c]) == color for r, c in expected):
        return "continuous"
    return "alternating"


def _find_containers(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    colors = sorted(int(c) for c in np.unique(grid) if int(c) not in {bg, YELLOW})
    containers = []
    for color in colors:
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        opening = _opening_from_bbox(grid, color, r0, c0, r1, c1)
        if opening is None:
            continue
        pattern = _detect_pattern(grid, color, r0, c0, r1, c1, opening)
        if pattern is None:
            continue
        containers.append(
            {
                "color": color,
                "opening": opening,
                "pattern": pattern,
                "bbox": (r0, c0, r1, c1),
            }
        )
    return containers


def _corridor_cells(shape: tuple[int, int], bbox: tuple[int, int, int, int], opening: str) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    if opening == "up":
        for r in range(0, r0):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif opening == "down":
        for r in range(r1 + 1, h):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif opening == "left":
        for r in range(r0 + 1, r1):
            for c in range(0, c0):
                cells.append((r, c))
    else:
        for r in range(r0 + 1, r1):
            for c in range(c1 + 1, w):
                cells.append((r, c))
    return cells


def _interior_fill_order(bbox: tuple[int, int, int, int], opening: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    if opening == "up":
        for r in range(r1 - 1, r0, -1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif opening == "down":
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif opening == "left":
        for c in range(c1 - 1, c0, -1):
            for r in range(r0 + 1, r1):
                cells.append((r, c))
    else:
        for c in range(c0 + 1, c1):
            for r in range(r0 + 1, r1):
                cells.append((r, c))
    return cells


def _fill_slots(bbox: tuple[int, int, int, int], opening: str, pattern: str) -> list[tuple[int, int]]:
    cells = _interior_fill_order(bbox, opening)
    if pattern == "continuous":
        return cells
    return [(r, c) for r, c in cells if (r + c) % 2 == 0]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for container in _find_containers(g):
        opening = container["opening"]
        pattern = container["pattern"]
        bbox = container["bbox"]
        yellow_pts = [(r, c) for r, c in _corridor_cells(g.shape, bbox, opening) if int(g[r, c]) == YELLOW]
        for r, c in yellow_pts:
            out[r, c] = BG
        slots = _fill_slots(bbox, opening, pattern)
        if len(yellow_pts) > len(slots):
            raise AssertionError("too many yellow pixels for container capacity")
        for r, c in slots[: len(yellow_pts)]:
            out[r, c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _reserve_bounds(
    r0: int,
    c0: int,
    height: int,
    width: int,
    opening: str,
    corridor_depth: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = shape
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    rr0, cc0, rr1, cc1 = r0, c0, r1, c1
    if opening == "up":
        rr0 = max(0, r0 - corridor_depth)
    elif opening == "down":
        rr1 = min(h - 1, r1 + corridor_depth)
    elif opening == "left":
        cc0 = max(0, c0 - corridor_depth)
    else:
        cc1 = min(w - 1, c1 + corridor_depth)
    return max(0, rr0 - 1), max(0, cc0 - 1), min(h - 1, rr1 + 1), min(w - 1, cc1 + 1)


def _parity_group(cells: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in cells:
        key = (cell[0] % 2, cell[1] % 2)
        groups.setdefault(key, []).append(cell)
    best_len = max(len(group) for group in groups.values())
    best_groups = [group for group in groups.values() if len(group) == best_len]
    return list(rng.choice(best_groups))


def _sample_yellows(
    bbox: tuple[int, int, int, int],
    opening: str,
    pattern: str,
    corridor_depth: int,
    shape: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    if opening == "up":
        for r in range(r0 - corridor_depth, r0):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif opening == "down":
        for r in range(r1 + 1, r1 + corridor_depth + 1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif opening == "left":
        for r in range(r0 + 1, r1):
            for c in range(c0 - corridor_depth, c0):
                cells.append((r, c))
    else:
        for r in range(r0 + 1, r1):
            for c in range(c1 + 1, c1 + corridor_depth + 1):
                cells.append((r, c))
    if not cells:
        return None
    allowed = _parity_group(cells, rng)
    rng.shuffle(allowed)
    interior_area = (r1 - r0 - 1) * (c1 - c0 - 1)
    fill_capacity = len(_fill_slots(bbox, opening, pattern))
    max_count = min(len(allowed), max(1, interior_area // 2), fill_capacity)
    if max_count <= 0:
        return None
    min_count = 1 if max_count == 1 else 2 if max_count >= 2 and rng.random() < 0.65 else 1
    count = rng.randint(min_count, max_count)
    return allowed[:count]


def _place_container(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    opening: str,
    pattern: str,
    rng: random.Random,
) -> dict | None:
    h, w = grid.shape
    for _ in range(150):
        height = rng.randint(*OUTER_H_RANGE)
        width = rng.randint(*OUTER_W_RANGE)
        if height >= h - 2 or width >= w - 2:
            continue
        corridor_depth = rng.randint(*CORRIDOR_DEPTH_RANGE)
        if opening == "up":
            r0_min, r0_max = corridor_depth + 1, h - height - 1
            c0_min, c0_max = 1, w - width - 1
        elif opening == "down":
            r0_min, r0_max = 1, h - height - corridor_depth - 2
            c0_min, c0_max = 1, w - width - 1
        elif opening == "left":
            r0_min, r0_max = 1, h - height - 1
            c0_min, c0_max = corridor_depth + 1, w - width - 1
        else:
            r0_min, r0_max = 1, h - height - 1
            c0_min, c0_max = 1, w - width - corridor_depth - 2
        if r0_min > r0_max or c0_min > c0_max:
            continue
        r0 = rng.randint(r0_min, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        reserve = _reserve_bounds(r0, c0, height, width, opening, corridor_depth, grid.shape)
        rr0, cc0, rr1, cc1 = reserve
        if np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            continue
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        yellow_pts = _sample_yellows(bbox, opening, pattern, corridor_depth, grid.shape, rng)
        if not yellow_pts:
            continue
        mask = _container_mask(height, width, opening, pattern)
        patch = grid[r0 : r0 + height, c0 : c0 + width]
        if np.any(mask & (patch != BG)):
            continue
        patch[mask] = color
        for r, c in yellow_pts:
            grid[r, c] = YELLOW
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return {
            "color": color,
            "opening": opening,
            "pattern": pattern,
            "bbox": bbox,
            "yellow_count": len(yellow_pts),
        }
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    area = h * w
    max_count = min(CONTAINER_COUNT_RANGE[1], max(1, area // 95))
    count = rng.randint(CONTAINER_COUNT_RANGE[0], max_count)
    colors = rng.sample(CONTAINER_COLORS, count)

    containers = []
    opening_pool = list(OPENINGS)
    pattern_pool = list(PATTERNS)
    rng.shuffle(opening_pool)
    rng.shuffle(pattern_pool)
    for idx in range(count):
        opening = opening_pool[idx % len(opening_pool)] if rng.random() < 0.7 else rng.choice(OPENINGS)
        pattern = pattern_pool[idx % len(pattern_pool)] if rng.random() < 0.7 else rng.choice(PATTERNS)
        placed = _place_container(grid, occupied, colors[idx], opening, pattern, rng)
        if placed is None:
            return None
        containers.append(placed)

    found = _find_containers(grid)
    if len(found) != len(containers):
        return None
    if any(int((grid == YELLOW).sum()) == 0 for _ in [0]):
        return None
    try:
        out = _solve(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    for placed in containers:
        bbox = placed["bbox"]
        slots = _fill_slots(bbox, placed["opening"], placed["pattern"])
        if placed["yellow_count"] > len(slots):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
