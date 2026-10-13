from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
ORIENTATIONS = ("up", "down", "left", "right")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _container_pattern(orientation: str, depth: int) -> dict[str, object]:
    walls: set[tuple[int, int]] = set()
    if orientation == "up":
        for r in range(depth):
            walls.add((r, 0))
            walls.add((r, 2))
        for c in range(3):
            walls.add((depth, c))
        inside_from_deep = [(r, 1) for r in range(depth - 1, -1, -1)]
        mouth = (-1, 1)
        outward = (-1, 0)
        shape = (depth + 1, 3)
    elif orientation == "down":
        for c in range(3):
            walls.add((0, c))
        for r in range(1, depth + 1):
            walls.add((r, 0))
            walls.add((r, 2))
        inside_from_deep = [(r, 1) for r in range(1, depth + 1)]
        mouth = (depth + 1, 1)
        outward = (1, 0)
        shape = (depth + 1, 3)
    elif orientation == "left":
        for c in range(depth):
            walls.add((0, c))
            walls.add((2, c))
        for r in range(3):
            walls.add((r, depth))
        inside_from_deep = [(1, c) for c in range(depth - 1, -1, -1)]
        mouth = (1, -1)
        outward = (0, -1)
        shape = (3, depth + 1)
    elif orientation == "right":
        for c in range(1, depth + 1):
            walls.add((0, c))
            walls.add((2, c))
        for r in range(3):
            walls.add((r, 0))
        inside_from_deep = [(1, c) for c in range(1, depth + 1)]
        mouth = (1, depth + 1)
        outward = (0, 1)
        shape = (3, depth + 1)
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    return {
        "walls": walls,
        "inside_from_deep": inside_from_deep,
        "mouth": mouth,
        "outward": outward,
        "shape": shape,
    }


def _recognize_container(comp: list[tuple[int, int]]) -> dict[str, object] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    norm = {(r - r0, c - c0) for r, c in comp}

    matches: list[tuple[str, int, dict[str, object]]] = []
    if w == 3 and 4 <= h <= 6:
        depth = h - 1
        for orientation in ("up", "down"):
            pattern = _container_pattern(orientation, depth)
            if norm == pattern["walls"]:
                matches.append((orientation, depth, pattern))
    if h == 3 and 4 <= w <= 6:
        depth = w - 1
        for orientation in ("left", "right"):
            pattern = _container_pattern(orientation, depth)
            if norm == pattern["walls"]:
                matches.append((orientation, depth, pattern))
    if len(matches) != 1:
        return None

    orientation, depth, pattern = matches[0]
    inside = [(r0 + dr, c0 + dc) for dr, dc in pattern["inside_from_deep"]]
    mouth = (r0 + int(pattern["mouth"][0]), c0 + int(pattern["mouth"][1]))
    return {
        "bbox": (r0, c0, r1, c1),
        "orientation": orientation,
        "depth": depth,
        "inside_from_deep": inside,
        "mouth": mouth,
        "outward": pattern["outward"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect bullets along each container's opening axis and stack them by arrival order.

    Input structure:
    - Black background.
    - Several gray U-shaped containers, each with a 1-pixel-wide interior corridor of depth 3-5.
    - Colored single-pixel bullets lie somewhere on the straight line extending from each opening.

    Rule:
    - Bullets travel toward the container along that line.
    - The nearest bullet arrives first and occupies the deepest interior cell.
    - Later bullets stack toward the opening; overflow bullets line up just outside the mouth.
    - Original bullet positions become black. Gray walls stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    containers: list[dict[str, object]] = []
    for comp in _components(g == WALL):
        info = _recognize_container(comp)
        if info is not None:
            containers.append(info)
    containers.sort(key=lambda item: tuple(item["bbox"]))

    for info in containers:
        mr, mc = info["mouth"]
        dr, dc = info["outward"]

        bullets: list[int] = []
        bullet_cells: list[tuple[int, int]] = []
        r = mr
        c = mc
        outside_slots: list[tuple[int, int]] = []
        while 0 <= r < h and 0 <= c < w:
            outside_slots.append((int(r), int(c)))
            val = int(g[r, c])
            if val not in (BG, WALL):
                bullets.append(val)
                bullet_cells.append((int(r), int(c)))
            r += dr
            c += dc

        for r, c in bullet_cells:
            out[r, c] = BG

        fill_slots = list(info["inside_from_deep"]) + outside_slots
        for color, (r, c) in zip(bullets, fill_slots):
            out[r, c] = int(color)

    return out


# --------------- generator ---------------

MIN_LANE = 6
GRID_RANGE = (18, 30)
COUNT_OPTIONS = (1, 2, 3, 4)
COUNT_WEIGHTS = (1, 2, 2, 1)
PLACEMENT_RETRIES = 120
GEN_RETRIES = 400


def _reserve_box(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = reserved.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_colors(count: int) -> list[int]:
    if count == 1:
        return [random.choice(PALETTE)]
    distinct = random.randint(2, min(4, count, len(PALETTE)))
    palette = random.sample(PALETTE, distinct)
    return [random.choice(palette) for _ in range(count)]


def _origin_ranges(orientation: str, depth: int, h: int, w: int) -> tuple[range, range] | None:
    if orientation == "up":
        r0_lo = max(h // 2, MIN_LANE)
        r0_hi = h - (depth + 1) - 1
        c0_lo = 1
        c0_hi = w - 4
    elif orientation == "down":
        r0_lo = 1
        r0_hi = min(h // 2 - depth - 1, h - depth - MIN_LANE - 1)
        c0_lo = 1
        c0_hi = w - 4
    elif orientation == "left":
        r0_lo = 1
        r0_hi = h - 4
        c0_lo = max(w // 2, MIN_LANE)
        c0_hi = w - (depth + 1) - 1
    elif orientation == "right":
        r0_lo = 1
        r0_hi = h - 4
        c0_lo = 1
        c0_hi = min(w // 2 - depth - 1, w - depth - MIN_LANE - 1)
    else:
        raise ValueError(f"unknown orientation: {orientation}")

    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    return range(r0_lo, r0_hi + 1), range(c0_lo, c0_hi + 1)


def _place_container(
    grid: np.ndarray,
    reserved: np.ndarray,
    orientation: str,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        depth = random.randint(3, 5)
        ranges = _origin_ranges(orientation, depth, h, w)
        if ranges is None:
            continue
        row_range, col_range = ranges
        r0 = random.choice(list(row_range))
        c0 = random.choice(list(col_range))

        pattern = _container_pattern(orientation, depth)
        walls = [(r0 + dr, c0 + dc) for dr, dc in pattern["walls"]]
        inside = [(r0 + dr, c0 + dc) for dr, dc in pattern["inside_from_deep"]]
        mouth = (r0 + int(pattern["mouth"][0]), c0 + int(pattern["mouth"][1]))
        dr, dc = pattern["outward"]

        lane: list[tuple[int, int]] = []
        r = mouth[0]
        c = mouth[1]
        while 0 <= r < h and 0 <= c < w:
            lane.append((int(r), int(c)))
            r += dr
            c += dc
        lane_len = len(lane)
        if lane_len < MIN_LANE:
            continue

        rr = [r for r, _ in walls + inside]
        cc = [c for _, c in walls + inside]
        if reserved[max(0, min(rr) - 1) : min(h, max(rr) + 2), max(0, min(cc) - 1) : min(w, max(cc) + 2)].any():
            continue
        if any(reserved[r, c] for r, c in lane):
            continue

        overflow = random.random() < 0.5
        if overflow:
            max_count = min(depth + 3, lane_len)
            if max_count <= depth:
                continue
            bullet_count = random.randint(depth + 1, max_count)
        else:
            if random.random() < 0.5:
                bullet_count = depth
            else:
                bullet_count = random.randint(1, depth - 1)

        distance_pool = list(range(2, lane_len + 1))
        if len(distance_pool) < bullet_count:
            distance_pool = list(range(1, lane_len + 1))
        if len(distance_pool) < bullet_count:
            continue
        distances = sorted(random.sample(distance_pool, bullet_count))
        colors = _sample_colors(bullet_count)

        for r, c in walls:
            grid[r, c] = WALL
        for color, dist in zip(colors, distances):
            rr = mouth[0] + dr * (dist - 1)
            cc = mouth[1] + dc * (dist - 1)
            grid[rr, cc] = int(color)

        _reserve_box(reserved, min(rr for rr, _ in walls + inside) - 1, min(cc for _, cc in walls + inside) - 1,
                     max(rr for rr, _ in walls + inside) + 1, max(cc for _, cc in walls + inside) + 1)
        for r, c in lane:
            reserved[r, c] = True
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    count = random.choices(COUNT_OPTIONS, weights=COUNT_WEIGHTS, k=1)[0]
    orientations = random.sample(ORIENTATIONS, count)
    random.shuffle(orientations)

    for orientation in orientations:
        if not _place_container(grid, reserved, orientation):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(grid == WALL) and len(_components(grid == WALL)) != len(_components(out == WALL)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
