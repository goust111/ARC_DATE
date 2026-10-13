from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 7, 8)
PAIR_COUNTS = (3, 4)
GAP_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 400
SHAPE_BBOX_LIMIT = 4


Coord = tuple[int, int]
Box = tuple[int, int, int, int]
Shape = tuple[Coord, ...]


def _normalize_shape(coords: np.ndarray | list[Coord] | tuple[Coord, ...]) -> Shape:
    arr = np.asarray(coords, dtype=int)
    origin = arr.min(axis=0)
    shifted = arr - origin
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_dims(shape: Shape) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_signature(shape: Shape) -> tuple[int, int, int]:
    h, w = _shape_dims(shape)
    return len(shape), h, w


def _mirror_shape(shape: Shape) -> Shape:
    h, w = _shape_dims(shape)
    mirrored = [(r, w - 1 - c) for r, c in shape]
    return _normalize_shape(mirrored)


def _is_connected_shape(shape: Shape) -> bool:
    cells = set(shape)
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbr = (r + dr, c + dc)
            if nbr in cells and nbr not in seen:
                stack.append(nbr)
    return len(seen) == len(cells)


def _shape_patch(shape: Shape, color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _bbox_from_coords(coords: np.ndarray) -> Box:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            coords: list[Coord] = []
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _bbox_gap_ok(a: Box, b: Box, min_gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(ar0 - br1 - 1, br0 - ar1 - 1, 0)
    col_gap = max(ac0 - bc1 - 1, bc0 - ac1 - 1, 0)
    return max(row_gap, col_gap) >= int(min_gap)


def _bridge_info(upper: dict, lower: dict) -> tuple[list[Coord], int, int]:
    ur0, uc0, ur1, uc1 = map(int, upper["bbox"])
    lr0, lc0, lr1, lc1 = map(int, lower["bbox"])
    assert ur0 < lr0, "pair must have a unique upper sprite"

    if uc1 < lc0:
        start = (ur1, uc1)
        end = (lr0, lc0)
        direction = 1
    elif lc1 < uc0:
        start = (ur1, uc0)
        end = (lr0, lc1)
        direction = -1
    else:
        raise AssertionError("paired sprites must be diagonally separated")

    row_delta = int(end[0] - start[0])
    col_delta = int(end[1] - start[1])
    assert row_delta > 1, "bridge must contain at least one interior pixel"
    assert abs(row_delta) == abs(col_delta), "facing corners must align on a 45-degree diagonal"

    cells = [(start[0] + step, start[1] + direction * step) for step in range(1, row_delta)]
    return cells, int(upper["color"]), direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shaped differently colored sprites and draw the diagonal bridge between them.

    Input structure:
    - A black-background grid contains 6 or 8 small 4-connected monochrome sprites.
    - Each normalized sprite mask appears exactly twice, but the two copies have different colors.
    - Each matched pair is diagonally offset so one copy is above-left or above-right of the other.

    Rule:
    - Extract each sprite's bounding-box mask and normalize it by translation only.
    - Pair the two sprites with the same normalized shape.
    - For each pair, identify the upper sprite, take the facing lower corner of its bbox and the facing upper corner of the lower sprite's bbox, then draw the open 45-degree segment between them.
    - Keep the original sprites unchanged, and color the bridge with the upper sprite's color.

    Key insight:
    - Color is a distraction: the pairing is determined only by exact normalized shape equality, without rotation or reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _find_components(g, bg=BG)
    assert len(objects) in (6, 8), "expected 3 or 4 shape pairs"

    shape_to_objects: dict[Shape, list[dict]] = defaultdict(list)
    for obj in objects:
        shape = _normalize_shape(obj["coords"])
        shape_to_objects[shape].append(obj)

    out = g.copy()
    for pair in shape_to_objects.values():
        assert len(pair) == 2, "each normalized shape must appear exactly twice"
        assert pair[0]["color"] != pair[1]["color"], "paired sprites must have different colors"
        upper, lower = sorted(pair, key=lambda obj: int(obj["bbox"][0]))
        bridge_cells, bridge_color, _direction = _bridge_info(upper, lower)
        for r, c in bridge_cells:
            assert int(out[r, c]) == BG, "bridge path must be empty in the input"
            out[r, c] = bridge_color
    return out


def _build_shape_library() -> tuple[dict[int, list[Shape]], dict[Shape, Shape], dict[tuple[int, int, int], list[Shape]]]:
    cells = [(r, c) for r in range(SHAPE_BBOX_LIMIT) for c in range(SHAPE_BBOX_LIMIT)]
    seen: set[Shape] = set()
    shapes: list[Shape] = []

    for size in range(3, 8):
        for combo in itertools.combinations(cells, size):
            shape = _normalize_shape(combo)
            if shape in seen:
                continue
            seen.add(shape)
            h, w = _shape_dims(shape)
            if h < 2 or w < 2:
                continue
            if len(shape) == h * w:
                continue
            if not _is_connected_shape(shape):
                continue
            shapes.append(shape)

    shape_set = set(shapes)
    by_direction: dict[int, list[Shape]] = {1: [], -1: []}
    mirrors: dict[Shape, Shape] = {}
    by_signature: dict[tuple[int, int, int], list[Shape]] = defaultdict(list)

    for shape in shapes:
        h, w = _shape_dims(shape)
        cells_set = set(shape)
        if (0, 0) in cells_set and (h - 1, w - 1) in cells_set:
            by_direction[1].append(shape)
        if (0, w - 1) in cells_set and (h - 1, 0) in cells_set:
            by_direction[-1].append(shape)
        mirrored = _mirror_shape(shape)
        if mirrored in shape_set and mirrored != shape:
            mirrors[shape] = mirrored
        by_signature[_shape_signature(shape)].append(shape)

    assert by_direction[1] and by_direction[-1], "shape library must support both diagonal directions"
    assert any(shape in mirrors for shape in by_direction[1]), "need at least one mirror-confuser shape"
    return by_direction, mirrors, by_signature


SHAPES_BY_DIRECTION, SHAPE_MIRRORS, SHAPES_BY_SIGNATURE = _build_shape_library()
MIRROR_LR_SHAPES = [shape for shape in SHAPES_BY_DIRECTION[1] if shape in SHAPE_MIRRORS and SHAPE_MIRRORS[shape] in SHAPES_BY_DIRECTION[-1]]


def _choose_directions(pair_count: int) -> list[int]:
    directions = [1, -1]
    while len(directions) < pair_count:
        directions.append(random.choice((1, -1)))
    random.shuffle(directions)
    return directions


def _choose_shapes(directions: list[int]) -> list[Shape]:
    pair_count = len(directions)
    chosen: list[Shape | None] = [None] * pair_count
    used: set[Shape] = set()

    lr_slots = [idx for idx, direction in enumerate(directions) if direction == 1]
    rl_slots = [idx for idx, direction in enumerate(directions) if direction == -1]
    assert lr_slots and rl_slots

    mirror_base = random.choice(MIRROR_LR_SHAPES)
    lr_slot = random.choice(lr_slots)
    rl_slot = random.choice(rl_slots)
    chosen[lr_slot] = mirror_base
    chosen[rl_slot] = SHAPE_MIRRORS[mirror_base]
    used.add(mirror_base)
    used.add(SHAPE_MIRRORS[mirror_base])

    remaining = [idx for idx, shape in enumerate(chosen) if shape is None]
    if remaining:
        base_signature = _shape_signature(mirror_base)
        random.shuffle(remaining)
        for idx in remaining[:]:
            candidates = [
                shape
                for shape in SHAPES_BY_SIGNATURE[base_signature]
                if shape not in used and shape in SHAPES_BY_DIRECTION[directions[idx]]
            ]
            if candidates:
                picked = random.choice(candidates)
                chosen[idx] = picked
                used.add(picked)
                remaining.remove(idx)
                break

    for idx in remaining:
        candidates = [shape for shape in SHAPES_BY_DIRECTION[directions[idx]] if shape not in used]
        picked = random.choice(candidates)
        chosen[idx] = picked
        used.add(picked)

    return [shape for shape in chosen if shape is not None]


def _choose_color_pairs(pair_count: int) -> list[tuple[int, int]]:
    while True:
        pairs = [tuple(random.sample(PALETTE, 2)) for _ in range(pair_count)]
        flat = [color for pair in pairs for color in pair]
        if len(set(flat)) < len(flat):
            return [(int(a), int(b)) for a, b in pairs]


def _shape_cells(shape: Shape, top: int, left: int) -> list[Coord]:
    return [(top + r, left + c) for r, c in shape]


def _pair_candidate(shape: Shape, direction: int, gap: int, top: int, left: int, grid_shape: tuple[int, int]) -> dict | None:
    h, w = _shape_dims(shape)
    upper_bbox = (top, left, top + h - 1, left + w - 1)
    delta = gap + 1

    if direction == 1:
        lower_top = upper_bbox[2] + delta
        lower_left = upper_bbox[3] + delta
        start = (upper_bbox[2], upper_bbox[3])
        end = (lower_top, lower_left)
    else:
        lower_top = upper_bbox[2] + delta
        lower_left = left - delta - (w - 1)
        start = (upper_bbox[2], upper_bbox[1])
        end = (lower_top, lower_left + w - 1)

    lower_bbox = (lower_top, lower_left, lower_top + h - 1, lower_left + w - 1)
    height, width = grid_shape
    if min(*upper_bbox, *lower_bbox) < 0:
        return None
    if upper_bbox[2] >= height or upper_bbox[3] >= width:
        return None
    if lower_bbox[2] >= height or lower_bbox[3] >= width:
        return None

    bridge = [(start[0] + step, start[1] + direction * step) for step in range(1, gap + 1)]
    upper_cells = _shape_cells(shape, top, left)
    lower_cells = _shape_cells(shape, lower_top, lower_left)

    return {
        "upper_bbox": upper_bbox,
        "lower_bbox": lower_bbox,
        "upper_cells": upper_cells,
        "lower_cells": lower_cells,
        "bridge": bridge,
    }


def _valid_candidates(
    spec: dict,
    grid_shape: tuple[int, int],
    occupied: np.ndarray,
    placed_boxes: list[Box],
) -> list[dict]:
    shape = spec["shape"]
    direction = int(spec["direction"])
    gap = int(spec["gap"])
    h, w = _shape_dims(shape)
    height, width = grid_shape
    candidates: list[dict] = []

    for top in range(height - h + 1):
        for left in range(width - w + 1):
            candidate = _pair_candidate(shape, direction, gap, top, left, grid_shape)
            if candidate is None:
                continue
            pair_boxes = [candidate["upper_bbox"], candidate["lower_bbox"]]
            if any(not _bbox_gap_ok(box, old_box) for box in pair_boxes for old_box in placed_boxes):
                continue
            cells = candidate["upper_cells"] + candidate["lower_cells"] + candidate["bridge"]
            if any(bool(occupied[r, c]) for r, c in cells):
                continue
            candidates.append(candidate)

    random.shuffle(candidates)
    return candidates


def _paint_cells(grid: np.ndarray, cells: list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _validate_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    objects = _find_components(grid, bg=BG)
    assert len(objects) in (6, 8)
    assert all(3 <= int(obj["area"]) <= 7 for obj in objects)
    assert all(int(obj["color"]) in PALETTE for obj in objects)

    colors = [int(obj["color"]) for obj in objects]
    assert len(set(colors)) < len(colors), "generation should include color reuse across different shapes"

    shape_to_objects: dict[Shape, list[dict]] = defaultdict(list)
    for obj in objects:
        shape_to_objects[_normalize_shape(obj["coords"])].append(obj)
    assert len(shape_to_objects) * 2 == len(objects)
    assert all(len(pair) == 2 for pair in shape_to_objects.values())
    assert any(pair[0]["color"] != pair[1]["color"] for pair in shape_to_objects.values())
    assert all(pair[0]["color"] != pair[1]["color"] for pair in shape_to_objects.values())

    unique_shapes = list(shape_to_objects)
    signatures = Counter(_shape_signature(shape) for shape in unique_shapes)
    assert any(count >= 2 for count in signatures.values()), "need at least one same-size confuser family"
    assert any(_mirror_shape(shape) in shape_to_objects and _mirror_shape(shape) != shape for shape in unique_shapes), "need a mirror confuser pair"

    out = main(grid)
    diff = {tuple(coord) for coord in np.argwhere(out != grid)}
    expected_diff: set[Coord] = set()
    directions = set()
    for pair in shape_to_objects.values():
        upper, lower = sorted(pair, key=lambda obj: int(obj["bbox"][0]))
        bridge_cells, bridge_color, direction = _bridge_info(upper, lower)
        directions.add(direction)
        assert bridge_cells, "each pair must contribute a non-empty bridge"
        for r, c in bridge_cells:
            assert int(grid[r, c]) == BG
            assert int(out[r, c]) == bridge_color
            expected_diff.add((r, c))
    assert directions == {1, -1}, "generator should cover both diagonal directions within each sample"
    assert diff == expected_diff, "output may only add the intended bridge cells"


def _generation_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNTS)
    directions = _choose_directions(pair_count)
    shapes = _choose_shapes(directions)
    color_pairs = _choose_color_pairs(pair_count)
    grid_h = random.randint(14 + 2 * pair_count, min(22 + pair_count, 28))
    grid_w = random.randint(14 + 2 * pair_count, min(22 + pair_count, 28))

    specs = []
    for shape, direction, (upper_color, lower_color) in zip(shapes, directions, color_pairs):
        specs.append(
            {
                "shape": shape,
                "direction": direction,
                "gap": random.randint(*GAP_RANGE),
                "upper_color": upper_color,
                "lower_color": lower_color,
            }
        )

    order = list(range(pair_count))
    random.shuffle(order)
    order.sort(
        key=lambda idx: (
            _shape_signature(specs[idx]["shape"])[1] * _shape_signature(specs[idx]["shape"])[2],
            specs[idx]["gap"],
        ),
        reverse=True,
    )

    output = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed_boxes: list[Box] = []
    bridges: list[list[Coord]] = []

    for idx in order:
        candidates = _valid_candidates(specs[idx], output.shape, occupied, placed_boxes)
        if not candidates:
            return None
        chosen = random.choice(candidates[: min(len(candidates), 60)])

        _paint_cells(output, chosen["upper_cells"], specs[idx]["upper_color"])
        _paint_cells(output, chosen["lower_cells"], specs[idx]["lower_color"])
        _paint_cells(output, chosen["bridge"], specs[idx]["upper_color"])

        for r, c in chosen["upper_cells"] + chosen["lower_cells"] + chosen["bridge"]:
            occupied[r, c] = True
        placed_boxes.extend([chosen["upper_bbox"], chosen["lower_bbox"]])
        bridges.append(chosen["bridge"])

    inp = output.copy()
    for bridge in bridges:
        for r, c in bridge:
            inp[r, c] = BG

    try:
        _validate_input(inp)
        if np.array_equal(main(inp), output):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shape-pair diagonal-bridge input")
