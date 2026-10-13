import random
from collections import Counter, defaultdict, deque

import numpy as np


BG_COLOR = 8
COLORS = tuple(range(10))
GENERATION_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
PAIR_COUNT_RANGE = (2, 4)
PAIR_SHAPE_TRIES = 80
PLACEMENT_TRIES = 400
OBJECT_PADDING = 1

SHAPE_TEMPLATES = (
    (
        ".#.",
        "#M#",
        ".#.",
    ),
    (
        "###",
        "#M#",
        "###",
    ),
    (
        "###",
        ".M.",
        ".#.",
    ),
    (
        ".#.",
        ".M.",
        "###",
    ),
    (
        "#..",
        "#M#",
        "#..",
    ),
    (
        "..#",
        "#M#",
        "..#",
    ),
    (
        "###",
        ".M#",
        ".#.",
    ),
    (
        "###",
        "#M.",
        ".#.",
    ),
    (
        ".#.",
        ".M#",
        "###",
    ),
    (
        ".#.",
        "#M.",
        "###",
    ),
    (
        "##.",
        ".M#",
        ".##",
    ),
    (
        ".##",
        "#M.",
        "##.",
    ),
)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            queue = deque([(int(r), int(c))])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == int(bg):
                        continue
                    seen[nr, nc] = True
                    queue.append((int(nr), int(nc)))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            values = grid[coords[:, 0], coords[:, 1]]
            components.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "values": values,
                }
            )
    return components


def _decode_object(grid: np.ndarray, comp: dict) -> dict:
    counts = Counter(map(int, comp["values"]))
    assert len(counts) == 2, "each sprite must use one body color and one marker color"

    marker_colors = [color for color, n in counts.items() if int(n) == 1]
    assert len(marker_colors) == 1, "each sprite must contain exactly one marker pixel"
    marker_color = int(marker_colors[0])

    body_colors = [color for color, n in counts.items() if int(n) > 1]
    assert len(body_colors) == 1, "sprite body color must dominate the component"
    body_color = int(body_colors[0])

    marker_index = int(np.flatnonzero(comp["values"] == marker_color)[0])
    marker_r, marker_c = map(int, comp["coords"][marker_index])

    cells = [
        (int(r), int(c), int(grid[int(r), int(c)]))
        for r, c in comp["coords"]
    ]
    return {
        "bbox": comp["bbox"],
        "marker_color": marker_color,
        "body_color": body_color,
        "marker": (marker_r, marker_c),
        "cells": cells,
        "size": int(len(cells)),
    }


def _read_key(obj: dict) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = obj["bbox"]
    mr, mc = obj["marker"]
    return int(r0), int(c0), int(mr), int(mc)


def _compose_pair(a: dict, b: dict, bg: int) -> np.ndarray:
    ar, ac = a["marker"]
    br, bc = b["marker"]
    dr = int(ar - br)
    dc = int(ac - bc)

    painted: dict[tuple[int, int], int] = {}
    for r, c, color in b["cells"]:
        painted[(int(r) + dr, int(c) + dc)] = int(color)
    for r, c, _color in a["cells"]:
        painted[(int(r), int(c))] = int(a["body_color"])

    rows = [int(r) for r, _c in painted]
    cols = [int(c) for _r, c in painted]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for (r, c), color in painted.items():
        out[int(r) - r0, int(c) - c0] = int(color)
    return out


def _stack_crops(crops: list[np.ndarray], bg: int) -> np.ndarray:
    assert crops
    height = sum(int(crop.shape[0]) for crop in crops) + len(crops) - 1
    width = max(int(crop.shape[1]) for crop in crops)
    out = np.full((height, width), int(bg), dtype=int)

    top = 0
    for crop in crops:
        h, w = crop.shape
        out[top : top + h, :w] = crop
        top += int(h) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _mode_color(g)

    objects = [_decode_object(g, comp) for comp in _find_components(g, bg)]
    assert objects, "expected at least one sprite pair"

    grouped: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        grouped[int(obj["marker_color"])].append(obj)

    composites: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for members in grouped.values():
        assert len(members) == 2, "each marker color must appear in exactly two sprites"
        ordered = sorted(members, key=_read_key)
        anchor, shifted = ordered[0], ordered[1]
        composites.append((_read_key(anchor), _compose_pair(anchor, shifted, bg)))

    composites.sort(key=lambda item: item[0])
    return _stack_crops([crop for _key, crop in composites], bg)


def _template_array(rows: tuple[str, ...]) -> np.ndarray:
    arr = np.zeros((len(rows), len(rows[0])), dtype=int)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                arr[r, c] = 1
            elif ch == "M":
                arr[r, c] = 2
    return arr


def _trim_shape(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(arr > 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    trimmed = arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    marker = tuple(map(int, np.argwhere(trimmed == 2)[0]))
    return trimmed > 0, marker


def _random_shape() -> dict:
    arr = _template_array(random.choice(SHAPE_TEMPLATES))
    if random.random() < 0.5:
        arr = np.fliplr(arr)
    if random.random() < 0.5:
        arr = np.flipud(arr)
    arr = np.rot90(arr, k=random.randint(0, 3))
    mask, marker = _trim_shape(arr)
    h, w = mask.shape
    mr, mc = marker
    assert 0 < mr < h - 1 and 0 < mc < w - 1, "marker must remain embedded after transforms"
    return {
        "mask": mask,
        "marker": marker,
        "h": int(h),
        "w": int(w),
        "area": int(mask.sum()),
    }


def _shape_offsets(shape: dict) -> set[tuple[int, int]]:
    mr, mc = shape["marker"]
    return {
        (int(r) - int(mr), int(c) - int(mc))
        for r, c in np.argwhere(shape["mask"])
    }


def _overlap_count(shape_a: dict, shape_b: dict) -> int:
    return len(_shape_offsets(shape_a) & _shape_offsets(shape_b))


def _choose_pair_shapes(conflict_mode: bool) -> tuple[dict, dict]:
    fallback: tuple[dict, dict] | None = None
    for _ in range(PAIR_SHAPE_TRIES):
        shape_a = _random_shape()
        shape_b = _random_shape()
        overlap = _overlap_count(shape_a, shape_b)
        fallback = (shape_a, shape_b)
        if conflict_mode and overlap >= 4:
            return shape_a, shape_b
        if not conflict_mode and overlap <= 3:
            return shape_a, shape_b
    assert fallback is not None
    return fallback


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _try_place_objects(objects: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    placed_boxes: list[tuple[int, int, int, int]] = []

    order = sorted(range(len(objects)), key=lambda idx: -int(objects[idx]["shape"]["area"]))
    for idx in order:
        shape = objects[idx]["shape"]
        oh = int(shape["h"])
        ow = int(shape["w"])
        if oh > h or ow > w:
            return None

        candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(0, h - oh)
            left = random.randint(0, w - ow)
            bbox = (int(top), int(left), int(top + oh - 1), int(left + ow - 1))
            if all(_boxes_separate(bbox, other, OBJECT_PADDING) for other in placed_boxes):
                candidates.append((int(top), int(left), bbox))
                if len(candidates) >= 24:
                    break
        if not candidates:
            return None

        top, left, bbox = random.choice(candidates)
        placements[idx] = (int(top), int(left))
        placed_boxes.append(bbox)

    assert all(item is not None for item in placements)
    return [(int(top), int(left)) for top, left in placements if top is not None]


def _paint_object(grid: np.ndarray, obj: dict, top: int, left: int):
    shape = obj["shape"]
    for r, c in np.argwhere(shape["mask"]):
        grid[int(top) + int(r), int(left) + int(c)] = int(obj["body_color"])
    mr, mc = shape["marker"]
    grid[int(top) + int(mr), int(left) + int(mc)] = int(obj["marker_color"])


def _generate_objects() -> list[dict]:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    marker_colors = random.sample([c for c in COLORS if c != BG_COLOR], pair_count)
    body_pool = [c for c in COLORS if c not in marker_colors and c != BG_COLOR]

    objects: list[dict] = []
    for marker_color in marker_colors:
        conflict_mode = random.random() < 0.5
        shape_a, shape_b = _choose_pair_shapes(conflict_mode)

        if conflict_mode:
            body_a, body_b = random.sample(body_pool, 2)
        else:
            if random.random() < 0.7:
                body_a = body_b = random.choice(body_pool)
            else:
                body_a = random.choice(body_pool)
                others = [c for c in body_pool if c != body_a]
                body_b = random.choice(others) if others else body_a

        objects.append(
            {
                "shape": shape_a,
                "body_color": int(body_a),
                "marker_color": int(marker_color),
            }
        )
        objects.append(
            {
                "shape": shape_b,
                "body_color": int(body_b),
                "marker_color": int(marker_color),
            }
        )
    random.shuffle(objects)
    return objects


def _gen_attempt() -> np.ndarray | None:
    objects = _generate_objects()
    min_side = 12 + 3 * (len(objects) // 2)
    h = random.randint(max(GRID_HEIGHT_RANGE[0], min_side), GRID_HEIGHT_RANGE[1])
    w = random.randint(max(GRID_WIDTH_RANGE[0], min_side), GRID_WIDTH_RANGE[1])

    placements = _try_place_objects(objects, h, w)
    if placements is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for obj, (top, left) in zip(objects, placements):
        _paint_object(grid, obj, int(top), int(left))

    try:
        out = main(grid)
    except Exception:
        return None

    if not isinstance(out, np.ndarray):
        return None
    if int(np.count_nonzero(grid != BG_COLOR)) <= 0:
        return None
    if _mode_color(grid) != BG_COLOR:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
