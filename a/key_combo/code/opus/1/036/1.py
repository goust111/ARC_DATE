from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
GREY = 5
SNAKE_COLOR_POOL = [2, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
GRID_ROW_RANGE = (22, 29)
GRID_COL_RANGE = (22, 29)
PAIR_COUNT_CHOICES = [2, 3]
PAIR_COUNT_WEIGHTS = [0.55, 0.45]
AREA_TO_DIMS = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(5, 2)],
    12: [(3, 4), (4, 3)],
    15: [(5, 3)],
    16: [(4, 4)],
    20: [(5, 4)],
}
AREA_CHOICES = sorted(AREA_TO_DIMS)
BORDER_PROFILES = [
    (1, 2, 1, 2),
    (2, 1, 2, 1),
    (1, 2, 2, 1),
    (2, 1, 1, 2),
    (2, 2, 2, 2),
]
OBJECT_PADDING = 1
SNAKE_PADDING = 1
NOISE_PADDING = 1
NOISE_COUNT_RANGE = (3, 7)
MAX_GENERATION_ATTEMPTS = 500
MAX_PATH_ATTEMPTS = 300
MAX_SEQUENCE_ATTEMPTS = 300
MAX_PLACEMENT_ATTEMPTS = 120


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _hole_records(grid: np.ndarray) -> list[dict]:
    holes = []
    for comp in _components(grid == BLUE):
        r0, c0, r1, c1 = _bbox(comp)
        local = grid[r0 : r1 + 1, c0 : c1 + 1]
        for local_hole in _components(local == BG):
            if (
                np.any(local_hole[:, 0] == 0)
                or np.any(local_hole[:, 1] == 0)
                or np.any(local_hole[:, 0] == local.shape[0] - 1)
                or np.any(local_hole[:, 1] == local.shape[1] - 1)
            ):
                continue
            hole = local_hole + np.array([r0, c0])
            hr0, hc0, hr1, hc1 = _bbox(hole)
            hh = hr1 - hr0 + 1
            hw = hc1 - hc0 + 1
            assert int(hole.shape[0]) == hh * hw, "expected rectangular holes"
            holes.append(
                {
                    "coords": hole,
                    "bbox": (hr0, hc0, hr1, hc1),
                    "top_left": (hr0, hc0),
                    "height": hh,
                    "width": hw,
                    "area": int(hole.shape[0]),
                }
            )
    assert holes, "expected at least one enclosed hole"
    assert len({hole["area"] for hole in holes}) == len(holes), "hole areas must be unique"
    return holes


def _neighbor_count(cell: tuple[int, int], cell_set: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in cell_set for dr, dc in DIRS4)


def _order_path(cell_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    degrees = {cell: _neighbor_count(cell, cell_set) for cell in cell_set}
    endpoints = sorted(cell for cell, deg in degrees.items() if deg == 1)
    assert len(endpoints) == 2, "snake must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees.values()), "snake must be an unbranched chain"

    start = endpoints[0]
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = []
        for dr, dc in DIRS4:
            cand = (cur[0] + dr, cur[1] + dc)
            if cand in cell_set and cand != prev:
                nxts.append(cand)
        if not nxts:
            break
        assert len(nxts) == 1, "snake path must be unique"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(cell_set)
    return order


def _snake_and_noise_records(grid: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    snakes = []
    noises = []
    for comp in _components((grid != BG) & (grid != BLUE)):
        cells = {tuple(map(int, rc)) for rc in comp.tolist()}
        colors = {int(grid[r, c]) for r, c in cells}
        if len(cells) == 1:
            only = next(iter(cells))
            assert int(grid[only]) == GREY, "singleton non-blue components must be grey noise"
            noises.append(comp)
            continue
        order = _order_path(cells)
        snakes.append(
            {
                "coords": comp,
                "order": order,
                "area": len(order),
                "colors": [int(grid[r, c]) for r, c in order],
                "palette": colors,
            }
        )
    assert snakes, "expected at least one snake path"
    assert len({snake["area"] for snake in snakes}) == len(snakes), "snake lengths must be unique"
    return snakes, noises


def main(grid: np.ndarray) -> np.ndarray:
    """Fold each multicolor snake path into the equal-area enclosed rectangular hole of a blue object, then erase the snakes and grey markers.

    Input structure:
    - The grid has black background, 2-3 separated blue 4-connected objects, and each blue object contains one enclosed rectangular black hole.
    - Separate from those objects are 2-3 multicolor non-blue snake paths: each is one 4-connected unbranched chain with exactly two endpoints.
    - Additional grey singleton pixels may appear as isolated marker noise away from both the blue objects and the snakes.

    Rule:
    - Detect every enclosed rectangular hole inside the blue objects and record its area.
    - Detect every snake path among the non-black, non-blue components and pair each snake to the unique hole with the same number of cells.
    - For each snake, choose the endpoint with smaller row index, breaking ties by smaller column index, and traverse the unique path to read a 1D color sequence.
    - Fill the matched hole in row-major order with that sequence, then erase every original snake pixel and every grey noise singleton while leaving the blue objects unchanged.

    Key insight:
    - The snake is a 1D encoding of a 2D pattern: only the endpoint tie-break, the path traversal order, and the hole's own row-major frame together determine the correct fold-back filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    holes = _hole_records(g)
    snakes, noises = _snake_and_noise_records(g)

    hole_by_area = {hole["area"]: hole for hole in holes}
    snake_areas = {snake["area"] for snake in snakes}
    assert snake_areas == set(hole_by_area), "snake/hole areas must match exactly"

    out = g.copy()
    for snake in snakes:
        coords = snake["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for noise in noises:
        out[noise[:, 0], noise[:, 1]] = BG

    for snake in snakes:
        hole = hole_by_area[snake["area"]]
        colors = snake["colors"]
        hr0, hc0, hr1, hc1 = hole["bbox"]
        idx = 0
        for r in range(hr0, hr1 + 1):
            for c in range(hc0, hc1 + 1):
                out[r, c] = colors[idx]
                idx += 1
    return out


def _build_object_patch(hole_h: int, hole_w: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    top, bottom, left, right = random.choice(BORDER_PROFILES)
    patch = np.full((hole_h + top + bottom, hole_w + left + right), BLUE, dtype=int)
    patch[top : top + hole_h, left : left + hole_w] = BG
    return patch, (top, left, hole_h, hole_w)


def _possible_positions(
    reserved: np.ndarray, item_h: int, item_w: int, padding: int
) -> list[tuple[int, int]]:
    h, w = reserved.shape
    out = []
    for r in range(h - item_h + 1):
        for c in range(w - item_w + 1):
            r0 = max(0, r - padding)
            r1 = min(h, r + item_h + padding)
            c0 = max(0, c - padding)
            c1 = min(w, c + item_w + padding)
            if not reserved[r0:r1, c0:c1].any():
                out.append((r, c))
    return out


def _mark_reserved(reserved: np.ndarray, r: int, c: int, item_h: int, item_w: int, padding: int) -> None:
    h, w = reserved.shape
    r0 = max(0, r - padding)
    r1 = min(h, r + item_h + padding)
    c0 = max(0, c - padding)
    c1 = min(w, c + item_w + padding)
    reserved[r0:r1, c0:c1] = True


def _reflect(coords: list[tuple[int, int]], flip_rows: bool, flip_cols: bool) -> list[tuple[int, int]]:
    out = []
    for r, c in coords:
        out.append(((-r if flip_rows else r), (-c if flip_cols else c)))
    return out


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    return [tuple(map(int, rc)) for rc in arr.tolist()]


def _same_row_path(length: int) -> list[tuple[int, int]] | None:
    choices = []
    for depth in range(1, length):
        span = length - (2 * depth) - 1
        if span >= 2:
            choices.append((depth, span))
    if not choices:
        return None
    depth, span = random.choice(choices)
    r, c = 0, 0
    coords = [(r, c)]
    for _ in range(depth):
        r += 1
        coords.append((r, c))
    for _ in range(span):
        c += 1
        coords.append((r, c))
    for _ in range(depth):
        r -= 1
        coords.append((r, c))
    return coords


def _different_row_path(length: int) -> list[tuple[int, int]]:
    r, c = 0, length
    coords = [(r, c)]
    for step in range(length - 1):
        if step % 2 == 0:
            r += 1
        else:
            c -= 1
        coords.append((r, c))
    return coords


def _generate_path_cells(length: int, mode: str) -> list[tuple[int, int]] | None:
    for _ in range(80):
        if mode == "same_row":
            base = _same_row_path(length)
        elif mode == "different_row":
            base = _different_row_path(length)
        else:
            base = random.choice([_same_row_path(length), _different_row_path(length)])
        if base is None:
            continue
        coords = _normalize(_reflect(base, bool(random.getrandbits(1)), bool(random.getrandbits(1))))
        ordered = _order_path(set(coords))
        if ordered == sorted(ordered) or ordered == sorted(ordered)[::-1]:
            continue
        col_sorted = sorted(ordered, key=lambda rc: (rc[1], rc[0]))
        if ordered == col_sorted or ordered == col_sorted[::-1]:
            continue
        endpoints = sorted([ordered[0], ordered[-1]])
        same_row = endpoints[0][0] == endpoints[1][0]
        if mode == "same_row" and not same_row:
            continue
        if mode == "different_row" and same_row:
            continue
        return ordered
    return None


def _alt_read_orders(order: list[tuple[int, int]]) -> list[list[int]]:
    index = {cell: i for i, cell in enumerate(order)}
    coords = list(order)
    alt_coords = [
        list(reversed(coords)),
        sorted(coords),
        sorted(coords, reverse=True),
        sorted(coords, key=lambda rc: (rc[1], rc[0])),
        sorted(coords, key=lambda rc: (rc[1], rc[0]), reverse=True),
    ]
    perms = []
    seen = set()
    for alt in alt_coords:
        perm = tuple(index[cell] for cell in alt)
        if perm != tuple(range(len(order))) and perm not in seen:
            perms.append(list(perm))
            seen.add(perm)
    return perms


def _generate_sequence(
    order: list[tuple[int, int]],
    hole_h: int,
    hole_w: int,
    force_grey: bool,
) -> list[int] | None:
    length = len(order)
    perms = _alt_read_orders(order)
    for _ in range(MAX_SEQUENCE_ATTEMPTS):
        palette_size = min(len(SNAKE_COLOR_POOL), random.randint(3, min(5, len(SNAKE_COLOR_POOL))))
        palette = random.sample(SNAKE_COLOR_POOL, palette_size)
        if force_grey and GREY not in palette:
            palette[random.randrange(len(palette))] = GREY
        seq = [random.choice(palette) for _ in range(length)]
        if len(set(seq)) < 2:
            continue
        if force_grey and GREY not in seq:
            seq[random.randrange(length)] = GREY
        if seq == list(reversed(seq)):
            continue
        if any([seq[i] for i in perm] == seq for perm in perms):
            continue
        matrix = np.array(seq, dtype=int).reshape(hole_h, hole_w)
        if np.array_equal(matrix, matrix[::-1, ::-1]):
            continue
        return seq
    return None


def _translate(order: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(r + top, c + left) for r, c in order]


def _paint_object(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    view[patch == BLUE] = BLUE


def _oracle_output(grid: np.ndarray, objects: list[dict], snakes: list[dict], noises: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for snake in snakes:
        for r, c in snake["abs_order"]:
            out[r, c] = BG
    for r, c in noises:
        out[r, c] = BG
    hole_by_area = {obj["area"]: obj for obj in objects}
    for snake in snakes:
        obj = hole_by_area[snake["length"]]
        hr, hc = obj["hole_top_left"]
        idx = 0
        for r in range(hr, hr + obj["hole_h"]):
            for c in range(hc, hc + obj["hole_w"]):
                out[r, c] = snake["sequence"][idx]
                idx += 1
    return out


def _nontrivial_pairing(objects: list[dict], snakes: list[dict]) -> bool:
    hole_area_order = [
        obj["area"]
        for obj in sorted(objects, key=lambda obj: (obj["hole_top_left"][0], obj["hole_top_left"][1]))
    ]
    snake_area_order = [
        snake["length"]
        for snake in sorted(snakes, key=lambda snake: snake["bbox_top_left"])
    ]
    return hole_area_order != snake_area_order


def _place_scene(objects_spec: list[dict], snakes_spec: list[dict], grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    object_records = []
    snake_records = []

    object_order = sorted(objects_spec, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)
    for obj in object_order:
        patch = obj["patch"]
        positions = _possible_positions(reserved, patch.shape[0], patch.shape[1], OBJECT_PADDING)
        if not positions:
            return None
        top, left = random.choice(positions)
        _mark_reserved(reserved, top, left, patch.shape[0], patch.shape[1], OBJECT_PADDING)
        _paint_object(grid, patch, top, left)
        hole_r, hole_c, hole_h, hole_w = obj["hole_rect"]
        object_records.append(
            {
                "area": obj["area"],
                "hole_h": hole_h,
                "hole_w": hole_w,
                "hole_top_left": (top + hole_r, left + hole_c),
            }
        )

    snake_order = sorted(snakes_spec, key=lambda item: item["bbox_shape"][0] * item["bbox_shape"][1], reverse=True)
    for snake in snake_order:
        sh, sw = snake["bbox_shape"]
        positions = _possible_positions(reserved, sh, sw, SNAKE_PADDING)
        if not positions:
            return None
        top, left = random.choice(positions)
        _mark_reserved(reserved, top, left, sh, sw, SNAKE_PADDING)
        abs_order = _translate(snake["order"], top, left)
        for (r, c), color in zip(abs_order, snake["sequence"]):
            grid[r, c] = color
        bbox_top_left = (top, left)
        snake_records.append(
            {
                "length": snake["length"],
                "sequence": list(snake["sequence"]),
                "abs_order": abs_order,
                "bbox_top_left": bbox_top_left,
            }
        )

    if not _nontrivial_pairing(object_records, snake_records):
        return None

    noise_count = random.randint(*NOISE_COUNT_RANGE)
    noises = []
    for _ in range(noise_count):
        positions = _possible_positions(reserved, 1, 1, NOISE_PADDING)
        if not positions:
            return None
        top, left = random.choice(positions)
        _mark_reserved(reserved, top, left, 1, 1, NOISE_PADDING)
        grid[top, left] = GREY
        noises.append((top, left))

    expected = _oracle_output(grid, object_records, snake_records, noises)
    return grid, expected


def _make_specs() -> tuple[list[dict], list[dict]] | None:
    pair_count = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    areas = random.sample(AREA_CHOICES, pair_count)

    objects = []
    hole_dims_by_area = {}
    for area in areas:
        hole_h, hole_w = random.choice(AREA_TO_DIMS[area])
        patch, hole_rect = _build_object_patch(hole_h, hole_w)
        objects.append({"area": area, "patch": patch, "hole_rect": hole_rect})
        hole_dims_by_area[area] = (hole_h, hole_w)

    modes = ["same_row", "different_row"]
    while len(modes) < pair_count:
        modes.append(random.choice(["same_row", "different_row", "any"]))
    random.shuffle(modes)

    force_grey_index = random.randrange(pair_count) if random.random() < 0.55 else None
    snakes = []
    for idx, area in enumerate(areas):
        order = None
        for _ in range(MAX_PATH_ATTEMPTS):
            order = _generate_path_cells(area, modes[idx])
            if order is not None:
                break
        if order is None:
            return None
        arr = np.array(order, dtype=int)
        bbox = _bbox(arr)
        hole_h, hole_w = hole_dims_by_area[area]
        seq = _generate_sequence(
            order,
            hole_h=hole_h,
            hole_w=hole_w,
            force_grey=idx == force_grey_index,
        )
        if seq is None:
            return None
        snakes.append(
            {
                "length": area,
                "order": order,
                "bbox_shape": (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1),
                "sequence": seq,
            }
        )
    return objects, snakes


def _gen_attempt() -> np.ndarray | None:
    specs = _make_specs()
    if specs is None:
        return None
    objects, snakes = specs

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        h = random.randint(*GRID_ROW_RANGE)
        w = random.randint(*GRID_COL_RANGE)
        placed = _place_scene(objects, snakes, h, w)
        if placed is None:
            continue
        grid, expected = placed
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-fold hole-fill input")
