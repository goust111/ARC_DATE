import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(range(10))
ROOM_COUNT_CHOICES = (2, 3, 4)
ROOM_SIDE_MIN = 4
ROOM_SIDE_MAX = 6
PAIR_COUNT_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 400

BASE_SHAPES = {
    2: [
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ],
    3: [
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
    ],
    4: [
        ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3)),
        ((0, 0), (0, 1), (0, 2), (1, 2), (1, 3)),
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0)),
    ],
}


def _mode_color(values: np.ndarray) -> int:
    vals = [int(v) for v in values.reshape(-1)]
    return Counter(vals).most_common(1)[0][0]


def _bg_color(grid: np.ndarray) -> int:
    return _mode_color(np.asarray(grid, dtype=int))


def _normalize_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _rotate_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_coords(tuple((c, -r) for r, c in coords))


def _flip_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_coords(tuple((r, -c) for r, c in coords))


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    height = max(r for r, _ in coords) + 1
    width = max(c for _, c in coords) + 1
    return height, width


def _build_shape_variants() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    variants: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for long_side, bases in BASE_SHAPES.items():
        seen = set()
        shapes = []
        for base in bases:
            base_norm = _normalize_coords(base)
            for flipped in (base_norm, _flip_coords(base_norm)):
                cur = flipped
                for _ in range(4):
                    cur = _normalize_coords(cur)
                    if cur not in seen:
                        seen.add(cur)
                        shapes.append(cur)
                    cur = _rotate_coords(cur)
        variants[long_side] = shapes
    return variants


SHAPE_VARIANTS = _build_shape_variants()


def _component_from_coords(coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return {
        "coords": tuple(sorted(coords)),
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
    }


def _components_from_mask(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(_component_from_coords(coords))
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    return _components_from_mask(np.asarray(grid, dtype=int) == int(color))


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in grid.reshape(-1) if int(v) != int(bg)]
    assert colors, "expected at least one non-background color"
    return Counter(colors).most_common(1)[0][0]


def _lock_descriptor(grid: np.ndarray, wall: int, coords: tuple[tuple[int, int], ...]) -> dict | None:
    h, w = grid.shape
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})

    if len(rows) == 1:
        r = rows[0]
        c0, c1 = min(cols), max(cols)
        length = len(coords)
        if length != c1 - c0 + 1 or not (2 <= length <= 4):
            return None
        if r <= 0 or r >= h - 1 or c0 <= 0 or c1 >= w - 1:
            return None
        if int(grid[r, c0 - 1]) != wall or int(grid[r, c1 + 1]) != wall:
            return None
        for c in range(c0, c1 + 1):
            if int(grid[r - 1, c]) == wall or int(grid[r + 1, c]) == wall:
                return None
        return {"orientation": "horizontal", "length": length, "coords": coords}

    if len(cols) == 1:
        c = cols[0]
        r0, r1 = min(rows), max(rows)
        length = len(coords)
        if length != r1 - r0 + 1 or not (2 <= length <= 4):
            return None
        if c <= 0 or c >= w - 1 or r0 <= 0 or r1 >= h - 1:
            return None
        if int(grid[r0 - 1, c]) != wall or int(grid[r1 + 1, c]) != wall:
            return None
        for r in range(r0, r1 + 1):
            if int(grid[r, c - 1]) == wall or int(grid[r, c + 1]) == wall:
                return None
        return {"orientation": "vertical", "length": length, "coords": coords}

    return None


def _label_regions(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    labels = np.full((h, w), -1, dtype=int)
    label = 0
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or labels[r, c] != -1:
                continue
            queue = deque([(r, c)])
            labels[r, c] = label
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            label += 1
    return labels


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    wall = _infer_wall_color(g, bg)

    pairs = []
    lock_cells = set()
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {bg, wall}):
        components = _color_components(g, color)
        assert len(components) == 2, "each color must form exactly one lock and one key"
        lock = None
        key = None
        for comp in components:
            desc = _lock_descriptor(g, wall, comp["coords"])
            if desc is not None:
                assert lock is None, "each color must have a unique lock"
                lock = desc
            else:
                assert key is None, "each color must have a unique key"
                key = comp
        assert lock is not None and key is not None
        r0, c0, r1, c1 = key["bbox"]
        pair = {
            "color": color,
            "lock": lock,
            "key": key,
            "key_long_side": max(r1 - r0 + 1, c1 - c0 + 1),
        }
        pairs.append(pair)
        lock_cells.update(lock["coords"])

    barrier = g == int(wall)
    for r, c in lock_cells:
        barrier[r, c] = True
    labels = _label_regions(~barrier)

    for pair in pairs:
        kr, kc = pair["key"]["coords"][0]
        pair["key_label"] = int(labels[kr, kc])
        assert pair["key_label"] >= 0

    return {"grid": g, "bg": bg, "wall": wall, "labels": labels, "pairs": pairs}


def _side_labels(lock: dict, labels: np.ndarray) -> tuple[int, int]:
    first = set()
    second = set()
    if lock["orientation"] == "horizontal":
        for r, c in lock["coords"]:
            top = int(labels[r - 1, c])
            bottom = int(labels[r + 1, c])
            if top >= 0:
                first.add(top)
            if bottom >= 0:
                second.add(bottom)
    else:
        for r, c in lock["coords"]:
            left = int(labels[r, c - 1])
            right = int(labels[r, c + 1])
            if left >= 0:
                first.add(left)
            if right >= 0:
                second.add(right)
    assert len(first) == 1 and len(second) == 1
    a = next(iter(first))
    b = next(iter(second))
    assert a != b
    return a, b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the same-colored lock on a wall.

    A lock opens only when its pixel length equals the longer side of the key's
    bounding box. All qualifying locks are removed together. For every opened
    lock, fill the original region on the opposite side from the key with the
    key color, but only over background pixels.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    g = scene["grid"]
    bg = scene["bg"]
    labels = scene["labels"]
    out = g.copy()

    opened = []
    for pair in scene["pairs"]:
        if pair["lock"]["length"] == pair["key_long_side"]:
            opened.append(pair)
            for r, c in pair["lock"]["coords"]:
                out[r, c] = bg

    for pair in opened:
        a, b = _side_labels(pair["lock"], labels)
        if pair["key_label"] == a:
            target = b
        else:
            assert pair["key_label"] == b
            target = a
        fill_mask = (labels == target) & (g == bg)
        out[fill_mask] = int(pair["color"])

    return out


def _build_axis_bounds(sizes: list[int]) -> list[tuple[int, int]]:
    bounds = []
    cursor = 0
    for size in sizes:
        start = cursor + 1
        end = start + size - 1
        bounds.append((start, end))
        cursor = end + 1
    return bounds


def _build_rooms(
    row_sizes: list[int], col_sizes: list[int]
) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    row_bounds = _build_axis_bounds(row_sizes)
    col_bounds = _build_axis_bounds(col_sizes)
    rooms = {}
    for rr, (r0, r1) in enumerate(row_bounds):
        for cc, (c0, c1) in enumerate(col_bounds):
            rooms[(rr, cc)] = (r0, r1, c0, c1)
    return rooms, row_bounds, col_bounds


def _draw_walls(
    grid: np.ndarray, wall: int, row_bounds: list[tuple[int, int]], col_bounds: list[tuple[int, int]]
) -> None:
    grid[0, :] = wall
    grid[-1, :] = wall
    grid[:, 0] = wall
    grid[:, -1] = wall
    for r0, r1 in row_bounds[:-1]:
        grid[r1 + 1, :] = wall
    for c0, c1 in col_bounds[:-1]:
        grid[:, c1 + 1] = wall


def _candidate_edges(
    row_sizes: list[int], col_sizes: list[int], row_bounds: list[tuple[int, int]], col_bounds: list[tuple[int, int]]
) -> list[dict]:
    edges = []
    for rr, room_h in enumerate(row_sizes):
        for cc in range(len(col_sizes) - 1):
            c0, c1 = col_bounds[cc]
            edges.append(
                {
                    "orientation": "vertical",
                    "a": (rr, cc),
                    "b": (rr, cc + 1),
                    "span": room_h,
                    "line": c1 + 1,
                    "range_start": row_bounds[rr][0],
                }
            )
    for rr in range(len(row_sizes) - 1):
        for cc, room_w in enumerate(col_sizes):
            r0, r1 = row_bounds[rr]
            edges.append(
                {
                    "orientation": "horizontal",
                    "a": (rr, cc),
                    "b": (rr + 1, cc),
                    "span": room_w,
                    "line": r1 + 1,
                    "range_start": col_bounds[cc][0],
                }
            )
    return [edge for edge in edges if edge["span"] >= 4]


def _choose_disjoint_edges(edges: list[dict], count: int, rng: random.Random) -> list[dict] | None:
    for _ in range(60):
        shuffled = edges[:]
        rng.shuffle(shuffled)
        selected = []
        used_rooms = set()
        for edge in shuffled:
            if edge["a"] in used_rooms or edge["b"] in used_rooms:
                continue
            selected.append(edge)
            used_rooms.add(edge["a"])
            used_rooms.add(edge["b"])
            if len(selected) == count:
                return selected
    return None


def _choose_shape(long_side: int, room_h: int, room_w: int, rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    choices = []
    for shape in SHAPE_VARIANTS[long_side]:
        shape_h, shape_w = _shape_dims(shape)
        if shape_h <= room_h and shape_w <= room_w:
            choices.append(shape)
    if not choices:
        return None
    return rng.choice(choices)


def _place_key(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    color: int,
    long_side: int,
    rng: random.Random,
) -> tuple[tuple[int, int], ...] | None:
    r0, r1, c0, c1 = room
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    shape = _choose_shape(long_side, room_h, room_w, rng)
    if shape is None:
        return None

    shape_h, shape_w = _shape_dims(shape)
    extra_r = room_h - shape_h
    extra_c = room_w - shape_w
    margin_r = 1 if extra_r >= 2 else 0
    margin_c = 1 if extra_c >= 2 else 0
    top_min = r0 + margin_r
    top_max = r0 + extra_r - margin_r
    left_min = c0 + margin_c
    left_max = c0 + extra_c - margin_c
    top = rng.randint(top_min, top_max)
    left = rng.randint(left_min, left_max)

    placed = []
    for dr, dc in shape:
        rr = top + dr
        cc = left + dc
        assert grid[rr, cc] != color
        grid[rr, cc] = color
        placed.append((rr, cc))
    return tuple(sorted(placed))


def _place_lock(edge: dict, color: int, grid: np.ndarray, rng: random.Random) -> tuple[tuple[int, int], ...]:
    max_len = min(4, edge["span"] - 2)
    length = rng.randint(2, max_len)
    offset = rng.randint(1, edge["span"] - length - 1)
    coords = []
    if edge["orientation"] == "vertical":
        start_r = edge["range_start"] + offset
        for i in range(length):
            coords.append((start_r + i, edge["line"]))
    else:
        start_c = edge["range_start"] + offset
        for i in range(length):
            coords.append((edge["line"], start_c + i))
    for r, c in coords:
        grid[r, c] = color
    return tuple(coords)


def _expected_output(
    grid: np.ndarray,
    bg: int,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    pairs: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for pair in pairs:
        if pair["opens"]:
            for r, c in pair["lock_coords"]:
                out[r, c] = bg
    for pair in pairs:
        if not pair["opens"]:
            continue
        r0, r1, c0, c1 = rooms[pair["target_room"]]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == bg] = int(pair["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random
    row_count = rng.choice(ROOM_COUNT_CHOICES)
    col_count = rng.choice(ROOM_COUNT_CHOICES)
    row_sizes = [rng.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(row_count)]
    col_sizes = [rng.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(col_count)]

    rooms, row_bounds, col_bounds = _build_rooms(row_sizes, col_sizes)
    height = row_bounds[-1][1] + 2
    width = col_bounds[-1][1] + 2
    if height > 30 or width > 30:
        return None

    edges = _candidate_edges(row_sizes, col_sizes, row_bounds, col_bounds)
    max_pairs = min(3, len(edges), (row_count * col_count) // 2)
    possible_counts = [count for count in PAIR_COUNT_CHOICES if count <= max_pairs]
    if not possible_counts:
        return None
    pair_count = rng.choice(possible_counts)
    chosen_edges = _choose_disjoint_edges(edges, pair_count, rng)
    if chosen_edges is None:
        return None

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    bg = colors[0]
    wall = colors[1]
    pair_colors = colors[2 : 2 + pair_count]

    grid = np.full((height, width), bg, dtype=int)
    _draw_walls(grid, wall, row_bounds, col_bounds)

    if pair_count == 2:
        open_indices = {rng.randrange(2)}
    else:
        open_total = rng.choice((1, 2))
        open_indices = set(rng.sample(range(pair_count), open_total))

    pair_meta = []
    for idx, (edge, color) in enumerate(zip(chosen_edges, pair_colors)):
        lock_coords = _place_lock(edge, color, grid, rng)
        key_room, target_room = (edge["a"], edge["b"]) if rng.random() < 0.5 else (edge["b"], edge["a"])
        desired_long = len(lock_coords) if idx in open_indices else rng.choice([v for v in (2, 3, 4) if v != len(lock_coords)])
        key_coords = _place_key(grid, rooms[key_room], color, desired_long, rng)
        if key_coords is None:
            return None
        pair_meta.append(
            {
                "color": color,
                "lock_coords": lock_coords,
                "key_coords": key_coords,
                "key_room": key_room,
                "target_room": target_room,
                "opens": idx in open_indices,
            }
        )

    if _bg_color(grid) != bg:
        return None
    if _infer_wall_color(grid, bg) != wall:
        return None

    expected = _expected_output(grid, bg, rooms, pair_meta)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    colors_present = {int(v) for v in np.unique(expected)}
    if len(colors_present) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock puzzle")
