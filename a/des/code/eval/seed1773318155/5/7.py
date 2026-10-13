import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS = ("up", "down", "left", "right")
STAMP_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_MIN = 22
GRID_SIDE_MAX = 30
PLACEMENT_MARGIN = 1
CONTAINER_COUNT_CHOICES = (2, 3, 4)


def _most_common_color(grid: np.ndarray) -> int:
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
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, r1 - r0 + 1, c1 - c0 + 1


def _rotation_keys(coords: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    key, h, w = _normalize(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in key:
        mask[r, c] = True

    keys: list[tuple[tuple[int, int], ...]] = []
    cur = mask
    for _ in range(4):
        pts = tuple(sorted(tuple(map(int, p)) for p in np.argwhere(cur)))
        if pts not in keys:
            keys.append(pts)
        cur = np.rot90(cur)
    return keys


def _canonical_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return min(_rotation_keys(coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _rect_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = PLACEMENT_MARGIN
) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _blank_canvas_shape(direction: str, outer_h: int, outer_w: int) -> tuple[int, int]:
    if direction in ("up", "down"):
        return outer_h - 3, outer_w - 2
    return outer_h - 2, outer_w - 3


def _map_blank_coord(direction: str, r: int, c: int) -> tuple[int, int]:
    if direction == "up":
        return 1 + r, 1 + c
    if direction == "down":
        return 2 + r, 1 + c
    if direction == "left":
        return 1 + r, 1 + c
    if direction == "right":
        return 1 + r, 2 + c
    raise ValueError(direction)


def _stripe_positions(direction: str, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    if direction == "up":
        return [(outer_h - 2, c) for c in range(1, outer_w - 1)]
    if direction == "down":
        return [(1, c) for c in range(1, outer_w - 1)]
    if direction == "left":
        return [(r, outer_w - 2) for r in range(1, outer_h - 1)]
    if direction == "right":
        return [(r, 1) for r in range(1, outer_h - 1)]
    raise ValueError(direction)


def _available_positions(direction: str, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    canvas_h, canvas_w = _blank_canvas_shape(direction, outer_h, outer_w)
    return [_map_blank_coord(direction, r, c) for r in range(canvas_h) for c in range(canvas_w)]


def _random_connected_shape(h: int, w: int) -> list[tuple[int, int]] | None:
    area = h * w
    if area < 4:
        return None

    lo = max(2, int(np.ceil(area * 0.3)))
    hi = min(area - 1, int(np.floor(area * 0.75)))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    start = (random.randrange(h), random.randrange(w))
    shape = {start}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        cr, cc = cell
        for dr, dc in DIRS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in shape:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(shape) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in shape:
            continue
        shape.add(cell)
        add_frontier(cell)

    coords = sorted(shape)
    if len(coords) != target:
        return None
    if len(_rotation_keys(coords)) < 2:
        return None
    return coords


def _rotate_shape_key(shape: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    for _ in range(turns % 4):
        _, h, _ = _normalize(pts)
        pts = [(c, h - 1 - r) for r, c in pts]
    norm, _, _ = _normalize(pts)
    return norm


def _build_container_tile(
    bg: int,
    direction: str,
    outer_h: int,
    outer_w: int,
    body_color: int,
    stripe_color: int,
    blank_shape: list[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    tile = np.full((outer_h, outer_w), bg, dtype=int)

    if direction == "up":
        tile[:, 0] = body_color
        tile[:, -1] = body_color
        tile[-1, :] = body_color
    elif direction == "down":
        tile[:, 0] = body_color
        tile[:, -1] = body_color
        tile[0, :] = body_color
    elif direction == "left":
        tile[0, :] = body_color
        tile[-1, :] = body_color
        tile[:, -1] = body_color
    elif direction == "right":
        tile[0, :] = body_color
        tile[-1, :] = body_color
        tile[:, 0] = body_color
    else:
        raise ValueError(direction)

    blank_rel = [_map_blank_coord(direction, r, c) for r, c in blank_shape]
    blank_set = set(blank_rel)
    for r, c in _available_positions(direction, outer_h, outer_w):
        if (r, c) not in blank_set:
            tile[r, c] = body_color
    for r, c in _stripe_positions(direction, outer_h, outer_w):
        tile[r, c] = stripe_color
    return tile, blank_rel


def _render_stamp_tile(
    bg: int, blank_shape: list[tuple[int, int]], turns: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    base_key, _, _ = _normalize(blank_shape)
    rotated = _rotate_shape_key(base_key, turns)
    h, w = _shape_dims(rotated)
    tile = np.full((h, w), bg, dtype=int)
    coords = list(rotated)
    for r, c in coords:
        tile[r, c] = STAMP_COLOR
    return tile, coords


def _detect_container(
    grid: np.ndarray, component: list[tuple[int, int]], color: int, bg: int
) -> dict | None:
    r0, c0, r1, c1 = _bbox(component)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in component:
        comp_mask[r - r0, c - c0] = True

    candidates = []
    if np.all(comp_mask[:, 0]) and np.all(comp_mask[:, -1]) and np.all(comp_mask[-1, :]) and not np.any(
        comp_mask[0, 1:-1]
    ):
        candidates.append("up")
    if np.all(comp_mask[:, 0]) and np.all(comp_mask[:, -1]) and np.all(comp_mask[0, :]) and not np.any(
        comp_mask[-1, 1:-1]
    ):
        candidates.append("down")
    if np.all(comp_mask[0, :]) and np.all(comp_mask[-1, :]) and np.all(comp_mask[:, -1]) and not np.any(
        comp_mask[1:-1, 0]
    ):
        candidates.append("left")
    if np.all(comp_mask[0, :]) and np.all(comp_mask[-1, :]) and np.all(comp_mask[:, 0]) and not np.any(
        comp_mask[1:-1, -1]
    ):
        candidates.append("right")

    if len(candidates) != 1:
        return None
    direction = candidates[0]

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    stripe = _stripe_positions(direction, h, w)
    stripe_colors = {int(sub[r, c]) for r, c in stripe}
    if len(stripe_colors) != 1:
        return None
    stripe_color = stripe_colors.pop()
    if stripe_color in {bg, color, STAMP_COLOR}:
        return None

    interior_vals = {
        int(sub[r, c])
        for r in range(1, h - 1)
        for c in range(1, w - 1)
    }
    if not interior_vals.issubset({bg, color, stripe_color}):
        return None

    blank_abs = [
        (r0 + r, c0 + c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if int(sub[r, c]) == bg
    ]
    if not blank_abs:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "body_color": color,
        "stripe_color": stripe_color,
        "blank_abs": blank_abs,
        "shape_key": _normalize(blank_abs)[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each container's internal blank shape with the matching rotated stamp, recolored by the container's stripe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    containers = []
    for color in map(int, np.unique(g)):
        if color in {bg, STAMP_COLOR}:
            continue
        for component in _components(g == color):
            info = _detect_container(g, component, color, bg)
            if info is not None:
                containers.append(info)

    stamps = _components(g == STAMP_COLOR)
    assert containers, "expected at least one container"
    assert len(stamps) == len(containers), "expected one stamp per container"

    matched = set()
    for stamp in stamps:
        stamp_keys = set(_rotation_keys(stamp))
        candidates = [i for i, info in enumerate(containers) if info["shape_key"] in stamp_keys and i not in matched]
        assert len(candidates) == 1, "each stamp must match exactly one container"
        matched.add(candidates[0])

    assert len(matched) == len(containers)

    out = g.copy()
    for stamp in stamps:
        for r, c in stamp:
            out[r, c] = bg
    for idx in matched:
        info = containers[idx]
        for r, c in info["blank_abs"]:
            out[r, c] = info["stripe_color"]
    return out


def _choose_grid_shape(objects: list[dict], total_fg: int) -> tuple[int, int] | None:
    max_h = max(obj["tile"].shape[0] for obj in objects) + 2
    max_w = max(obj["tile"].shape[1] for obj in objects) + 2
    padded_area = sum((obj["tile"].shape[0] + 2) * (obj["tile"].shape[1] + 2) for obj in objects)
    min_area = max(total_fg * 2 + 20, padded_area + 10)

    candidates = []
    for h in range(GRID_SIDE_MIN, GRID_SIDE_MAX + 1):
        for w in range(GRID_SIDE_MIN, GRID_SIDE_MAX + 1):
            if h < max_h or w < max_w:
                continue
            if h * w < min_area:
                continue
            candidates.append((h, w))
    return random.choice(candidates) if candidates else None


def _place_objects(objects: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    rects: list[tuple[int, int, int, int]] = []
    ordered = sorted(objects, key=lambda obj: obj["tile"].shape[0] * obj["tile"].shape[1], reverse=True)

    for obj in ordered:
        tile_h, tile_w = obj["tile"].shape
        placed = False
        for _ in range(300):
            r0 = random.randint(1, h - tile_h - 1)
            c0 = random.randint(1, w - tile_w - 1)
            rect = (r0, c0, r0 + tile_h - 1, c0 + tile_w - 1)
            if all(_rect_separated(rect, other) for other in rects):
                rects.append(rect)
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    by_id = {id(obj): pos for obj, pos in zip(ordered, placements)}
    return [by_id[id(obj)] for obj in objects]


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != STAMP_COLOR])
    container_count = random.choice(CONTAINER_COUNT_CHOICES)

    palette = [c for c in range(10) if c not in {bg, STAMP_COLOR}]
    random.shuffle(palette)
    body_colors = palette[:container_count]
    stripe_colors = palette[container_count : 2 * container_count]
    directions = random.sample(DIRECTIONS, container_count)
    turns = random.sample([0, 1, 2, 3], container_count)

    objects = []
    used_shapes = set()

    for i in range(container_count):
        direction = directions[i]
        built = None
        for _ in range(80):
            outer_h = random.randint(6, 9)
            outer_w = random.randint(6, 9)
            canvas_h, canvas_w = _blank_canvas_shape(direction, outer_h, outer_w)
            blank_shape = _random_connected_shape(canvas_h, canvas_w)
            if blank_shape is None:
                continue
            canonical = _canonical_shape(blank_shape)
            if canonical in used_shapes:
                continue

            container_tile, blank_rel = _build_container_tile(
                bg,
                direction,
                outer_h,
                outer_w,
                body_colors[i],
                stripe_colors[i],
                blank_shape,
            )
            stamp_tile, stamp_rel = _render_stamp_tile(bg, blank_shape, turns[i])
            built = (
                {
                    "kind": "container",
                    "tile": container_tile,
                    "blank_rel": blank_rel,
                    "stripe_color": stripe_colors[i],
                },
                {
                    "kind": "stamp",
                    "tile": stamp_tile,
                    "stamp_rel": stamp_rel,
                },
            )
            used_shapes.add(canonical)
            break
        if built is None:
            return None
        objects.extend(built)

    total_fg = sum(int(np.count_nonzero(obj["tile"] != bg)) for obj in objects)
    shape = _choose_grid_shape(objects, total_fg)
    if shape is None:
        return None
    h, w = shape

    placements = _place_objects(objects, h, w)
    if placements is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    expected = None
    stamp_abs_coords: list[list[tuple[int, int]]] = []
    container_fills: list[tuple[list[tuple[int, int]], int]] = []

    for obj, (r0, c0) in zip(objects, placements):
        tile = obj["tile"]
        mask = tile != bg
        grid[r0 : r0 + tile.shape[0], c0 : c0 + tile.shape[1]][mask] = tile[mask]
        if obj["kind"] == "stamp":
            stamp_abs_coords.append([(r0 + r, c0 + c) for r, c in obj["stamp_rel"]])
        else:
            container_fills.append(([(r0 + r, c0 + c) for r, c in obj["blank_rel"]], obj["stripe_color"]))

    expected = grid.copy()
    for coords in stamp_abs_coords:
        for r, c in coords:
            expected[r, c] = bg
    for coords, color in container_fills:
        for r, c in coords:
            expected[r, c] = color

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
