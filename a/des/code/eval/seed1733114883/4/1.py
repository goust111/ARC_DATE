import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_ORDER = ("up", "down", "left", "right")
STEP_BY_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

FRAME_COLOR = 5
PALETTE = tuple(range(10))
NON_GREY_COLORS = tuple(color for color in PALETTE if color != FRAME_COLOR)
OUTER_SIZES = (7, 9, 11, 13)
MAX_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(component) != expected:
        return None

    cells = set(component)
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return None
    for r, c in cells:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    return (int(r0), int(c0), int(r1), int(c1))


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int, int, int], int]]:
    frames: list[tuple[tuple[int, int, int, int], int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for component in _components(grid == color):
            bbox = _frame_bbox(component)
            if bbox is not None:
                frames.append((bbox, color))
    frames.sort(key=lambda item: item[0])
    return frames


def _seed_counts(coords: np.ndarray) -> dict[str, int]:
    rows = coords[:, 0].astype(float)
    cols = coords[:, 1].astype(float)
    center_r = (float(rows.min()) + float(rows.max())) / 2.0
    center_c = (float(cols.min()) + float(cols.max())) / 2.0
    return {
        "up": int(np.sum(rows < center_r)),
        "down": int(np.sum(rows > center_r)),
        "left": int(np.sum(cols < center_c)),
        "right": int(np.sum(cols > center_c)),
    }


def _seed_direction(coords: np.ndarray) -> str:
    counts = _seed_counts(coords)
    best = max(counts.values())
    choices = [direction for direction in DIR_ORDER if counts[direction] == best]
    if len(choices) == 1:
        return choices[0]

    rows = coords[:, 0].astype(float)
    cols = coords[:, 1].astype(float)
    center_r = (float(rows.min()) + float(rows.max())) / 2.0
    center_c = (float(cols.min()) + float(cols.max())) / 2.0
    mean_r = float(rows.mean())
    mean_c = float(cols.mean())
    bias = {
        "up": center_r - mean_r,
        "down": mean_r - center_r,
        "left": center_c - mean_c,
        "right": mean_c - center_c,
    }
    return max(choices, key=lambda direction: (bias[direction], -DIR_ORDER.index(direction)))


def _line_cells(
    bbox: tuple[int, int, int, int],
    direction: str,
    length: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "up":
        start_r, start_c = r0 - 1, mid_c
    elif direction == "down":
        start_r, start_c = r1 + 1, mid_c
    elif direction == "left":
        start_r, start_c = mid_r, c0 - 1
    else:
        start_r, start_c = mid_r, c1 + 1

    dr, dc = STEP_BY_DIR[direction]
    cells: list[tuple[int, int]] = []
    rr, cc = start_r, start_c
    for _ in range(length):
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    for bbox, frame_color in _find_frames(g, bg):
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != bg) & (interior != frame_color)
        if not seed_mask.any():
            continue

        seed_values = [int(value) for value in interior[seed_mask]]
        seed_color = Counter(seed_values).most_common(1)[0][0]
        local_coords = np.argwhere(interior == seed_color)
        if local_coords.size == 0:
            continue
        coords = local_coords + np.array([r0 + 1, c0 + 1], dtype=int)
        direction = _seed_direction(coords)
        for rr, cc in _line_cells(bbox, direction, int(coords.shape[0]), (h, w)):
            out[rr, cc] = seed_color

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)


def _make_seed(direction: str, size: int, rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(250):
        box_h = rng.randint(3, 5)
        box_w = rng.randint(3, 5)
        if box_h * box_w < size:
            continue

        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        while len(cells) < size:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < box_h and 0 <= nc < box_w):
                        continue
                    if (nr, nc) in cells or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
            if not frontier:
                break

            weights = []
            for nr, nc in frontier:
                if direction == "up":
                    side_bias = box_h - nr
                elif direction == "down":
                    side_bias = nr + 1
                elif direction == "left":
                    side_bias = box_w - nc
                else:
                    side_bias = nc + 1
                touches = sum((nr + dr, nc + dc) in cells for dr, dc in DIRS4)
                weights.append(3 * touches + side_bias)
            cells.add(rng.choices(frontier, weights=weights, k=1)[0])

        if len(cells) != size:
            continue

        norm = _normalize(cells)
        seed = np.array(sorted(norm), dtype=int)
        height, width = _shape_dims(norm)
        if height < 2 or width < 2:
            continue

        counts = _seed_counts(seed)
        if list(counts.values()).count(max(counts.values())) != 1:
            continue
        if _seed_direction(seed) != direction:
            continue
        return norm

    raise RuntimeError(f"failed to generate a clear {direction} seed of size {size}")


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((int(nr), int(nc)))
    return expanded


def _frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _expected_output(grid: np.ndarray, objects: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for obj in objects:
        bbox = obj["bbox"]
        direction = obj["direction"]
        color = obj["color"]
        length = obj["length"]
        for rr, cc in _line_cells(bbox, direction, length, out.shape):
            out[rr, cc] = color
    return out


def _random_directions(count: int, rng: random.Random) -> list[str]:
    if count <= 4:
        return rng.sample(list(DIR_ORDER), count)
    directions = list(DIR_ORDER)
    directions.extend(rng.choice(DIR_ORDER) for _ in range(count - 4))
    rng.shuffle(directions)
    return directions


def _random_clip_flags(count: int, rng: random.Random) -> list[bool]:
    true_count = count // 2
    if count % 2 == 1 and rng.random() < 0.5:
        true_count += 1
    flags = [True] * true_count + [False] * (count - true_count)
    rng.shuffle(flags)
    if count >= 2 and all(flags):
        flags[0] = False
    if count >= 2 and not any(flags):
        flags[0] = True
    return flags


def _place_object(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    color: int,
    direction: str,
    clip: bool,
    seed: set[tuple[int, int]],
    rng: random.Random,
) -> dict[str, object] | None:
    h, w = grid.shape
    seed_h, seed_w = _shape_dims(seed)
    height_choices = [value for value in OUTER_SIZES if value >= seed_h + 4 and value <= h]
    width_choices = [value for value in OUTER_SIZES if value >= seed_w + 4 and value <= w]
    if not height_choices or not width_choices:
        return None

    for _ in range(150):
        outer_h = rng.choice(height_choices)
        outer_w = rng.choice(width_choices)
        r0 = rng.randrange(0, h - outer_h + 1)
        c0 = rng.randrange(0, w - outer_w + 1)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        line = _line_cells(bbox, direction, len(seed), grid.shape)
        if not line:
            continue
        if clip and len(line) >= len(seed):
            continue
        if not clip and len(line) < len(seed):
            continue

        inner_h = outer_h - 2
        inner_w = outer_w - 2
        max_seed_r = inner_h - seed_h - 1
        max_seed_c = inner_w - seed_w - 1
        if max_seed_r < 1 or max_seed_c < 1:
            continue

        seed_r = rng.randint(1, max_seed_r)
        seed_c = rng.randint(1, max_seed_c)
        abs_seed = {
            (r0 + 1 + seed_r + sr, c0 + 1 + seed_c + sc)
            for sr, sc in seed
        }
        occupied = _frame_cells(bbox) | abs_seed | set(line)
        blocked = _expand(occupied, grid.shape, pad=1)
        if blocked & reserved:
            continue

        for rr, cc in _frame_cells(bbox):
            grid[rr, cc] = FRAME_COLOR
        for rr, cc in abs_seed:
            grid[rr, cc] = color
        reserved.update(blocked)
        return {
            "bbox": bbox,
            "direction": direction,
            "color": color,
            "length": len(seed),
        }
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    bg = rng.choice(NON_GREY_COLORS)
    grid = np.full((h, w), bg, dtype=int)

    max_objects = 4 if h * w < 650 else 5
    count = rng.randint(2, max_objects)
    colors = [color for color in PALETTE if color not in (bg, FRAME_COLOR)]
    rng.shuffle(colors)
    colors = colors[:count]
    directions = _random_directions(count, rng)
    clips = _random_clip_flags(count, rng)

    reserved: set[tuple[int, int]] = set()
    objects: list[dict[str, object]] = []
    for index in range(count):
        length = rng.randint(4, 6)
        seed = _make_seed(directions[index], length, rng)
        obj = _place_object(grid, reserved, colors[index], directions[index], clips[index], seed, rng)
        if obj is None:
            return None
        objects.append(obj)

    if get_bg(grid) != bg:
        return None

    expected = _expected_output(grid, objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
