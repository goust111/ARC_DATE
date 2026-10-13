import math
import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    scene = grid[:-1, :]
    mask = (scene != BG) & (scene != GREY)
    objects = []
    for cells in _components(mask):
        color = int(scene[cells[0][0], cells[0][1]])
        bbox = _bbox(cells)
        objects.append({"cells": cells, "color": color, "bbox": bbox})
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _has_vote(grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str) -> bool:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return r0 - 1 >= 0 and bool(np.any(grid[r0 - 1, c0 : c1 + 1] == GREY))
    if direction == "down":
        return r1 + 1 < grid.shape[0] - 1 and bool(np.any(grid[r1 + 1, c0 : c1 + 1] == GREY))
    if direction == "left":
        return c0 - 1 >= 0 and bool(np.any(grid[r0 : r1 + 1, c0 - 1] == GREY))
    return c1 + 1 < grid.shape[1] and bool(np.any(grid[r0 : r1 + 1, c1 + 1] == GREY))


def _missing_direction(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    missing = [name for name in DIRS if not _has_vote(grid, bbox, name)]
    assert len(missing) == 1, f"expected one missing vote direction, got {missing}"
    return missing[0]


def _shadow_shift(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    span = (r1 - r0 + 1) if direction in ("up", "down") else (c1 - c0 + 1)
    dr, dc = DIRS[direction]
    return dr * span, dc * span


def _distance_to_boundary(
    bbox: tuple[int, int, int, int], direction: str, height: int, width: int
) -> int:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    if direction == "up":
        return center_r + 1
    if direction == "down":
        return height - center_r
    if direction == "left":
        return center_c + 1
    return width - center_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored object toward its missing-vote direction and color its shadow via the bottom legend.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    legend = g[-1, :]

    for obj in _extract_objects(g):
        direction = _missing_direction(g, obj["bbox"])
        dr, dc = _shadow_shift(obj["bbox"], direction)
        dist = _distance_to_boundary(obj["bbox"], direction, h, w)
        shadow_color = int(legend[min(max(dist, 1), w) - 1])

        for r, c in obj["cells"]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w and out[rr, cc] == BG:
                out[rr, cc] = shadow_color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 18)
GRID_W_PAD_RANGE = (4, 12)
OBJECT_COUNT_RANGE = (2, 4)
DIM_CHOICES = ((3, 3), (3, 5), (5, 3), (5, 5), (3, 7), (7, 3), (5, 7), (7, 5), (7, 7))
SHAPE_FILL_RANGE = (0.45, 0.75)
CLIP_PROB = 0.35
MAX_OBJECT_TRIES = 120
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _random_connected_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    path_len = height + width - 1
    low = max(path_len, int(math.ceil(area * SHAPE_FILL_RANGE[0])))
    high = min(area - 1, int(math.floor(area * SHAPE_FILL_RANGE[1])))
    if low > high:
        return None

    for _ in range(80):
        target = random.randint(low, high)
        mask = np.zeros((height, width), dtype=bool)
        steps = ["D"] * (height - 1) + ["R"] * (width - 1)
        random.shuffle(steps)
        r = c = 0
        mask[r, c] = True
        for step in steps:
            if step == "D":
                r += 1
            else:
                c += 1
            mask[r, c] = True

        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for nr, nc in _neighbors4(int(rr), int(cc), height, width):
                if not bool(mask[nr, nc]):
                    frontier.add((int(nr), int(nc)))

        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            rr, cc = cell
            mask[rr, cc] = True
            for nr, nc in _neighbors4(rr, cc, height, width):
                if not bool(mask[nr, nc]):
                    frontier.add((int(nr), int(nc)))

        if int(mask.sum()) != target or bool(mask.all()):
            continue
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        return mask.copy()
    return None


def _random_shape() -> np.ndarray | None:
    dims = random.choice(DIM_CHOICES)
    return _random_connected_mask(*dims)


def _sample_legend(width: int) -> np.ndarray:
    for _ in range(60):
        row = np.array(random.choices(PALETTE, k=width), dtype=int)
        for i in range(1, width):
            if row[i] == row[i - 1] and random.random() < 0.7:
                choices = [c for c in PALETTE if c != int(row[i - 1])]
                row[i] = random.choice(choices)
        if len(set(map(int, row.tolist()))) >= min(4, len(PALETTE)):
            if MAROON not in row:
                row[random.randrange(width)] = MAROON
            return row
    row = np.resize(np.array(PALETTE, dtype=int), width)
    return row.copy()


def _sample_directions(count: int) -> list[str]:
    base = list(DIRS.keys())
    random.shuffle(base)
    chosen = base[: min(count, len(base))]
    while len(chosen) < count:
        chosen.append(random.choice(base))
    random.shuffle(chosen)
    return chosen


def _placement_ranges(
    height: int, width: int, obj_h: int, obj_w: int, direction: str, clip: bool
) -> tuple[int, int, int, int] | None:
    r_min = 1 if direction != "up" else 0
    r_max = height - obj_h - (2 if direction != "down" else 1)
    c_min = 1 if direction != "left" else 0
    c_max = width - obj_w - (1 if direction != "right" else 0)

    if direction == "up":
        if clip:
            r_min = max(r_min, 1)
            r_max = min(r_max, obj_h - 1)
        else:
            r_min = max(r_min, obj_h)
    elif direction == "down":
        r_max = min(r_max, height - 2 * obj_h - 1)
    elif direction == "left":
        if clip:
            c_min = max(c_min, 1)
            c_max = min(c_max, obj_w - 1)
        else:
            c_min = max(c_min, obj_w)
    else:  # right
        if clip:
            c_min = max(c_min, width - 2 * obj_w + 1)
            c_max = min(c_max, width - obj_w - 1)
        else:
            c_max = min(c_max, width - 2 * obj_w)

    if r_min > r_max or c_min > c_max:
        return None
    return r_min, r_max, c_min, c_max


def _absolute_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _vote_cells(
    bbox: tuple[int, int, int, int], direction: str
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    votes: list[tuple[int, int]] = []
    if direction != "up":
        votes.append((r0 - 1, random.randint(c0, c1)))
    if direction != "down":
        votes.append((r1 + 1, random.randint(c0, c1)))
    if direction != "left":
        votes.append((random.randint(r0, r1), c0 - 1))
    if direction != "right":
        votes.append((random.randint(r0, r1), c1 + 1))
    return votes


def _shadow_cells(
    cells: list[tuple[int, int]], bbox: tuple[int, int, int, int], direction: str, height: int, width: int
) -> list[tuple[int, int]]:
    dr, dc = _shadow_shift(bbox, direction)
    projected = []
    for r, c in cells:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height - 1 and 0 <= cc < width:
            projected.append((int(rr), int(cc)))
    return projected


def _expand(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    expanded.add((int(rr), int(cc)))
    return expanded


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    direction: str,
) -> bool:
    height, width = grid.shape
    obj_h, obj_w = mask.shape
    clip = direction in ("up", "left", "right") and random.random() < CLIP_PROB
    ranges = _placement_ranges(height, width, obj_h, obj_w, direction, clip)
    if ranges is None:
        return False
    r_min, r_max, c_min, c_max = ranges

    for _ in range(MAX_OBJECT_TRIES):
        top = random.randint(r_min, r_max)
        left = random.randint(c_min, c_max)
        bbox = (top, left, top + obj_h - 1, left + obj_w - 1)
        cells = _absolute_cells(mask, top, left)
        votes = _vote_cells(bbox, direction)
        shadow = _shadow_cells(cells, bbox, direction, height, width)
        if not shadow:
            continue
        if clip and len(shadow) == len(cells):
            continue
        if (not clip) and len(shadow) != len(cells):
            continue

        core = set(cells) | set(votes) | set(shadow)
        if any(bool(blocked[r, c]) for r, c in core):
            continue

        for r, c in cells:
            grid[r, c] = color
        for r, c in votes:
            grid[r, c] = GREY

        for r, c in _expand(core, height, width):
            blocked[r, c] = True
        return True
    return False


def _structural_check(grid: np.ndarray) -> bool:
    objects = _extract_objects(grid)
    if not (OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]):
        return False

    directions = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        if (r1 - r0) % 2 == 1 or (c1 - c0) % 2 == 1:
            return False
        try:
            directions.append(_missing_direction(grid, obj["bbox"]))
        except AssertionError:
            return False

    if len(set(directions)) < 2:
        return False
    if np.any(grid[-1, :] == BG) or np.any(grid[-1, :] == GREY):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(height, height + random.randint(*GRID_W_PAD_RANGE))

    grid = np.full((height, width), BG, dtype=int)
    grid[-1, :] = _sample_legend(width)
    blocked = np.zeros((height, width), dtype=bool)
    blocked[-1, :] = True

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, object_count)
    directions = _sample_directions(object_count)

    for color, direction in zip(colors, directions):
        mask = _random_shape()
        if mask is None:
            return None
        if not _place_object(grid, blocked, mask, int(color), direction):
            return None

    if not _structural_check(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[-1, :], grid[-1, :]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
