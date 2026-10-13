from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEFT = (0, -1)
RIGHT = (0, 1)
UP = (-1, 0)
DOWN = (1, 0)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for nxt in _neighbors4(cell):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _body_center(body: set[tuple[int, int]]) -> tuple[int, int] | None:
    rows = [r for r, _ in body]
    cols = [c for _, c in body]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if (r0 + r1) % 2 != 0 or (c0 + c1) % 2 != 0:
        return None
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return center if center in body else None


def _extract_object(cells: set[tuple[int, int]]) -> dict:
    degrees = {
        cell: sum((nxt in cells) for nxt in _neighbors4(cell))
        for cell in cells
    }
    tips = [cell for cell, degree in degrees.items() if degree == 1]
    candidates: list[dict] = []

    for tip in tips:
        neighbors = [nxt for nxt in _neighbors4(tip) if nxt in cells]
        if len(neighbors) != 1:
            continue
        first = neighbors[0]
        ray_dir = (first[0] - tip[0], first[1] - tip[1])
        tail_dir = (-ray_dir[0], -ray_dir[1])

        tail = [tip]
        prev = tip
        cur = first
        valid = True

        while True:
            next_cell = (cur[0] + ray_dir[0], cur[1] + ray_dir[1])
            cur_neighbors = [nxt for nxt in _neighbors4(cur) if nxt in cells]
            side_neighbors = [
                nxt for nxt in cur_neighbors if nxt not in {prev, next_cell}
            ]
            if side_neighbors:
                break
            if next_cell not in cells:
                valid = False
                break
            if set(cur_neighbors) != {prev, next_cell}:
                valid = False
                break
            tail.append(cur)
            prev, cur = cur, next_cell

        if not valid:
            continue

        tail_set = set(tail)
        body = cells - tail_set
        if not body or not _connected(body):
            continue

        attachments = {
            nxt
            for cell in tail_set
            for nxt in _neighbors4(cell)
            if nxt in body
        }
        if len(attachments) != 1:
            continue

        center = _body_center(body)
        if center is None:
            continue

        candidates.append(
            {
                "cells": cells,
                "body": body,
                "tail": tail_set,
                "center": center,
                "tail_dir": tail_dir,
                "ray_dir": ray_dir,
                "area": len(body),
            }
        )

    assert len(candidates) == 1, "each object must have one uniquely identifiable tail"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each tail, then cast a reverse-tail ray whose reach equals the body area.
    The first object touched by a ray is recolored to the emitter's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    objects: list[dict] = []
    cell_to_object: dict[tuple[int, int], int] = {}

    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        cells = set(coords)
        assert cells and _connected(cells), "each color must form one 4-connected object"
        obj = _extract_object(cells)
        obj["color"] = color
        obj["index"] = len(objects)
        for cell in cells:
            assert cell not in cell_to_object, "objects must not overlap"
            cell_to_object[cell] = obj["index"]
        objects.append(obj)

    incoming: dict[int, int] = {}
    ray_cells: dict[tuple[int, int], int] = {}

    for obj in objects:
        r, c = obj["center"]
        dr, dc = obj["ray_dir"]

        while (r + dr, c + dc) in obj["cells"]:
            r += dr
            c += dc
        nr, nc = r + dr, c + dc

        for _ in range(obj["area"]):
            if not (0 <= nr < height and 0 <= nc < width):
                break
            target = cell_to_object.get((nr, nc))
            if target is not None and target != obj["index"]:
                assert target not in incoming, "a target must not receive conflicting rays"
                incoming[target] = obj["index"]
                break
            prev = ray_cells.get((nr, nc))
            assert prev in (None, obj["color"]), "different-colored rays must not overlap"
            ray_cells[(nr, nc)] = obj["color"]
            nr += dr
            nc += dc

    out = np.full_like(g, BG)
    for obj in objects:
        paint_color = obj["color"]
        sender = incoming.get(obj["index"])
        if sender is not None:
            paint_color = objects[sender]["color"]
        for r, c in obj["body"]:
            out[r, c] = int(paint_color)

    for (r, c), color in ray_cells.items():
        assert out[r, c] in (BG, color), "rays must not overwrite object bodies"
        out[r, c] = int(color)

    return out


# --------------- generator ---------------

BODY_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))
TAIL_LEN_RANGE = (1, 2)
GAP_RANGE = (1, 3)
LANE_GAP_RANGE = (2, 4)
MARGIN_RANGE = (1, 3)
CONTENT_EXTRA_RANGE = (0, 3)
MAX_SIZE = 30
GEN_RETRIES = 400
LANE_PATTERNS = (
    (2,),
    (3,),
    (1, 2),
    (1, 3),
    (2, 2),
    (1, 1, 2),
    (1, 2, 2),
    (1, 1, 3),
)


def _render_object(
    grid: np.ndarray,
    center: tuple[int, int],
    shape: tuple[int, int],
    tail_dir: tuple[int, int],
    tail_len: int,
    color: int,
) -> None:
    cr, cc = center
    h, w = shape
    body_top = cr - h // 2
    body_left = cc - w // 2
    grid[body_top : body_top + h, body_left : body_left + w] = int(color)

    if tail_dir == UP:
        tip_r, tip_c = body_top, cc
        for step in range(1, tail_len + 1):
            grid[tip_r - step, tip_c] = int(color)
    elif tail_dir == DOWN:
        tip_r, tip_c = body_top + h - 1, cc
        for step in range(1, tail_len + 1):
            grid[tip_r + step, tip_c] = int(color)
    elif tail_dir == LEFT:
        tip_r, tip_c = cr, body_left
        for step in range(1, tail_len + 1):
            grid[tip_r, tip_c - step] = int(color)
    else:
        assert tail_dir == RIGHT
        tip_r, tip_c = cr, body_left + w - 1
        for step in range(1, tail_len + 1):
            grid[tip_r, tip_c + step] = int(color)


def _build_horizontal_lane(
    length: int,
    colors: list[int],
    ray_dir: tuple[int, int],
) -> dict:
    shapes = [random.choice(BODY_SHAPES) for _ in range(length)]
    tails = [random.randint(*TAIL_LEN_RANGE) for _ in range(length)]
    center_row = max(h for h, _ in shapes) // 2
    tail_dir = RIGHT if ray_dir == LEFT else LEFT

    objects = []
    if ray_dir == LEFT:
        body_left = 0
        prev_full_right = None
        for idx in range(length):
            h, w = shapes[idx]
            tail_len = tails[idx]
            center_col = body_left + w // 2
            body_right = body_left + w - 1
            full_right = body_right + tail_len
            objects.append(
                {
                    "center": (center_row, center_col),
                    "shape": (h, w),
                    "tail_dir": tail_dir,
                    "tail_len": tail_len,
                    "color": colors[idx],
                }
            )
            prev_full_right = full_right
            if idx + 1 < length:
                next_area = shapes[idx + 1][0] * shapes[idx + 1][1]
                gap = random.randint(1, min(GAP_RANGE[1], next_area - 1))
                body_left = prev_full_right + gap + 1
        width = prev_full_right + 1
    else:
        prev_body_right = None
        for idx in range(length):
            h, w = shapes[idx]
            tail_len = tails[idx]
            if idx == 0:
                body_left = tail_len
            else:
                prev_area = shapes[idx - 1][0] * shapes[idx - 1][1]
                gap = random.randint(1, min(GAP_RANGE[1], prev_area - 1))
                full_left = prev_body_right + gap + 1
                body_left = full_left + tail_len
            center_col = body_left + w // 2
            body_right = body_left + w - 1
            objects.append(
                {
                    "center": (center_row, center_col),
                    "shape": (h, w),
                    "tail_dir": tail_dir,
                    "tail_len": tail_len,
                    "color": colors[idx],
                }
            )
            prev_body_right = body_right
        width = prev_body_right + 1

    return {"objects": objects, "height": max(h for h, _ in shapes), "width": width}


def _build_vertical_lane(
    length: int,
    colors: list[int],
    ray_dir: tuple[int, int],
) -> dict:
    shapes = [random.choice(BODY_SHAPES) for _ in range(length)]
    tails = [random.randint(*TAIL_LEN_RANGE) for _ in range(length)]
    center_col = max(w for _, w in shapes) // 2
    tail_dir = DOWN if ray_dir == UP else UP

    objects = []
    if ray_dir == UP:
        body_top = 0
        prev_full_bottom = None
        for idx in range(length):
            h, w = shapes[idx]
            tail_len = tails[idx]
            center_row = body_top + h // 2
            body_bottom = body_top + h - 1
            full_bottom = body_bottom + tail_len
            objects.append(
                {
                    "center": (center_row, center_col),
                    "shape": (h, w),
                    "tail_dir": tail_dir,
                    "tail_len": tail_len,
                    "color": colors[idx],
                }
            )
            prev_full_bottom = full_bottom
            if idx + 1 < length:
                next_area = shapes[idx + 1][0] * shapes[idx + 1][1]
                gap = random.randint(1, min(GAP_RANGE[1], next_area - 1))
                body_top = prev_full_bottom + gap + 1
        height = prev_full_bottom + 1
    else:
        prev_body_bottom = None
        for idx in range(length):
            h, w = shapes[idx]
            tail_len = tails[idx]
            if idx == 0:
                body_top = tail_len
            else:
                prev_area = shapes[idx - 1][0] * shapes[idx - 1][1]
                gap = random.randint(1, min(GAP_RANGE[1], prev_area - 1))
                full_top = prev_body_bottom + gap + 1
                body_top = full_top + tail_len
            center_row = body_top + h // 2
            body_bottom = body_top + h - 1
            objects.append(
                {
                    "center": (center_row, center_col),
                    "shape": (h, w),
                    "tail_dir": tail_dir,
                    "tail_len": tail_len,
                    "color": colors[idx],
                }
            )
            prev_body_bottom = body_bottom
        height = prev_body_bottom + 1

    return {"objects": objects, "height": height, "width": max(w for _, w in shapes)}


def _pack_horizontal_lanes(lanes: list[dict]) -> dict | None:
    content_width = max(lane["width"] for lane in lanes) + random.randint(*CONTENT_EXTRA_RANGE)
    y = 0
    packed = []
    for lane in lanes:
        x = random.randint(0, content_width - lane["width"])
        packed.append({"lane": lane, "offset": (y, x)})
        y += lane["height"]
        if lane is not lanes[-1]:
            y += random.randint(*LANE_GAP_RANGE)

    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    height = y + top + bottom
    width = content_width + left + right
    if max(height, width) > MAX_SIZE:
        return None

    objects = []
    for item in packed:
        off_r, off_c = item["offset"]
        for obj in item["lane"]["objects"]:
            r, c = obj["center"]
            objects.append(
                {
                    **obj,
                    "center": (r + top + off_r, c + left + off_c),
                }
            )
    return {"grid_shape": (height, width), "objects": objects}


def _pack_vertical_lanes(lanes: list[dict]) -> dict | None:
    content_height = max(lane["height"] for lane in lanes) + random.randint(*CONTENT_EXTRA_RANGE)
    x = 0
    packed = []
    for lane in lanes:
        y = random.randint(0, content_height - lane["height"])
        packed.append({"lane": lane, "offset": (y, x)})
        x += lane["width"]
        if lane is not lanes[-1]:
            x += random.randint(*LANE_GAP_RANGE)

    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    height = content_height + top + bottom
    width = x + left + right
    if max(height, width) > MAX_SIZE:
        return None

    objects = []
    for item in packed:
        off_r, off_c = item["offset"]
        for obj in item["lane"]["objects"]:
            r, c = obj["center"]
            objects.append(
                {
                    **obj,
                    "center": (r + top + off_r, c + left + off_c),
                }
            )
    return {"grid_shape": (height, width), "objects": objects}


def _attempt_generate() -> np.ndarray | None:
    pattern = list(random.choice(LANE_PATTERNS))
    random.shuffle(pattern)
    total_objects = sum(pattern)
    colors = random.sample(range(1, 10), total_objects)
    orientation = random.choice(("H", "V"))

    lanes = []
    color_index = 0
    for length in pattern:
        lane_colors = colors[color_index : color_index + length]
        color_index += length
        if orientation == "H":
            ray_dir = random.choice((LEFT, RIGHT))
            lanes.append(_build_horizontal_lane(length, lane_colors, ray_dir))
        else:
            ray_dir = random.choice((UP, DOWN))
            lanes.append(_build_vertical_lane(length, lane_colors, ray_dir))

    packed = _pack_horizontal_lanes(lanes) if orientation == "H" else _pack_vertical_lanes(lanes)
    if packed is None:
        return None

    grid_shape = packed["grid_shape"]
    objects = packed["objects"]
    grid = np.full(grid_shape, BG, dtype=int)

    for obj in objects:
        _render_object(
            grid,
            obj["center"],
            obj["shape"],
            obj["tail_dir"],
            obj["tail_len"],
            obj["color"],
        )

    colors_present = [int(color) for color in np.unique(grid) if int(color) != BG]
    if len(colors_present) != total_objects:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
