from __future__ import annotations

from collections import deque
import random

import numpy as np


OPENINGS = ("up", "down", "left", "right")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
MAX_GENERATE_ATTEMPTS = 400
MAX_SCENE_TRIES = 140
MAX_EXTRA_FOOD_TRIES = 80

FOOD_BASE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
)


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    comps: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        mask = grid == color
        seen = np.zeros_like(mask, dtype=bool)
        for sr, sc in np.argwhere(mask):
            sr = int(sr)
            sc = int(sc)
            if seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in DIRS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append({"color": color, "cells": tuple(cells), "bbox": _bbox(cells)})
    return comps


def _container_mask(height: int, width: int, opening: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if opening == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif opening == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif opening == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    return mask


def _cavity_cells(bbox: tuple[int, int, int, int], opening: str) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = bbox
    if opening == "up":
        rows = range(r0, r1)
        cols = range(c0 + 1, c1)
    elif opening == "down":
        rows = range(r0 + 1, r1 + 1)
        cols = range(c0 + 1, c1)
    elif opening == "left":
        rows = range(r0 + 1, r1)
        cols = range(c0, c1)
    else:
        rows = range(r0 + 1, r1)
        cols = range(c0 + 1, c1 + 1)
    return tuple((r, c) for r in rows for c in cols)


def _classify_container(comp: dict[str, object]) -> dict[str, object] | None:
    cells = comp["cells"]
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    local = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        local[r - r0, c - c0] = True

    for opening in OPENINGS:
        if np.array_equal(local, _container_mask(height, width, opening)):
            info = dict(comp)
            info["opening"] = opening
            info["cavity"] = _cavity_cells((r0, c0, r1, c1), opening)
            return info
    return None


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _food_distance(container: dict[str, object], food: dict[str, object]) -> int | None:
    r0, c0, r1, c1 = container["bbox"]
    fr0, fc0, fr1, fc1 = food["bbox"]
    opening = container["opening"]
    if opening == "up":
        if fr1 >= r0 or not _overlap(c0 + 1, c1 - 1, fc0, fc1):
            return None
        return r0 - fr1
    if opening == "down":
        if fr0 <= r1 or not _overlap(c0 + 1, c1 - 1, fc0, fc1):
            return None
        return fr0 - r1
    if opening == "left":
        if fc1 >= c0 or not _overlap(r0 + 1, r1 - 1, fr0, fr1):
            return None
        return c0 - fc1
    if fc0 <= c1 or not _overlap(r0 + 1, r1 - 1, fr0, fr1):
        return None
    return fc0 - c1


def _nearest_food(container: dict[str, object], foods: list[dict[str, object]]) -> dict[str, object] | None:
    ranked: list[tuple[tuple[int, int, int, int, int], dict[str, object]]] = []
    for food in foods:
        distance = _food_distance(container, food)
        if distance is None:
            continue
        ranked.append(((distance, *food["bbox"], len(food["cells"])), food))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each three-sided container with the color of the nearest food object in its opening direction.

    Input structure:
    - Background is black. Foreground contains several monochrome three-sided containers and several small solid food objects.
    - Each container is a 1-pixel-thick U/C-shaped component whose tight bounding box reveals a single opening on one cardinal side.
    - Each food is a separate 4-connected monochrome block of 1 to 4 pixels, using colors different from the container walls.

    Rule:
    - Detect every container by matching its connected component to one of the four axis-aligned three-sided frame templates.
    - For each container, look only along its opening direction and consider food components whose bounding boxes overlap the opening span.
    - Choose the food with the smallest positive orthogonal distance to that opening edge, erase that food from the grid, and fill the container's interior cavity with the food color.
    - Leave container walls unchanged and keep foods that are not selected by any container.

    Key insight:
    - The same 3-sided frame geometry simultaneously encodes both the container's interior region and the direction of the ray used to choose its nearest food.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    containers: list[dict[str, object]] = []
    foods: list[dict[str, object]] = []
    for comp in _components(g, bg):
        container = _classify_container(comp)
        if container is None:
            foods.append(comp)
        else:
            containers.append(container)

    containers.sort(key=lambda comp: comp["bbox"])
    for container in containers:
        food = _nearest_food(container, foods)
        if food is None:
            continue
        for r, c in food["cells"]:
            out[r, c] = bg
        for r, c in container["cavity"]:
            out[r, c] = int(food["color"])

    return out


def _normalize_shape(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple((c, -r) for r, c in cells))


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple((r, -c) for r, c in cells))


def _random_food_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    shape = _normalize_shape(rng.choice(FOOD_BASE_SHAPES))
    for _ in range(rng.randrange(4)):
        shape = _rotate_shape(shape)
    if rng.random() < 0.5:
        shape = _flip_shape(shape)
    return shape


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shift_cells(shape: tuple[tuple[int, int], ...], top: int, left: int) -> tuple[tuple[int, int], ...]:
    return tuple((top + r, left + c) for r, c in shape)


def _scene_bbox(parts: list[tuple[tuple[int, int], ...]]) -> tuple[int, int, int, int]:
    cells = [cell for part in parts for cell in part]
    return _bbox(cells)


def _overlap_start(lo: int, hi: int, size: int, limit: int, rng: random.Random) -> int | None:
    start_lo = max(0, lo - size + 1)
    start_hi = min(limit - size, hi)
    if start_lo > start_hi:
        return None
    return rng.randint(start_lo, start_hi)


def _paint(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _place_scene(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    opening: str,
    container_color: int,
    food_colors: list[int],
    force_secondary: bool,
) -> dict[str, object] | None:
    h_grid, w_grid = grid.shape
    for _ in range(MAX_SCENE_TRIES):
        box_h = rng.randint(4, 8)
        box_w = rng.randint(4, 8)
        primary_shape = _random_food_shape(rng)
        food_h, food_w = _shape_dims(primary_shape)
        near_gap = rng.randint(1, 4)

        secondary_shape: tuple[tuple[int, int], ...] | None = None
        far_gap = 0
        if force_secondary or rng.random() < 0.3:
            secondary_shape = _random_food_shape(rng)
            far_gap = rng.randint(1, 3)
        if secondary_shape is not None:
            far_h, far_w = _shape_dims(secondary_shape)
        else:
            far_h = far_w = 0

        if opening == "up":
            top_need = food_h + near_gap + (far_h + far_gap if secondary_shape is not None else 0)
            if top_need > h_grid - box_h:
                continue
            r0 = rng.randint(top_need, h_grid - box_h)
            c0 = rng.randint(0, w_grid - box_w)
            food_top = r0 - near_gap - food_h
            food_left = _overlap_start(c0 + 1, c0 + box_w - 2, food_w, w_grid, rng)
            if food_left is None:
                continue
            far_top = None
            far_left = None
            if secondary_shape is not None:
                far_top = food_top - far_gap - far_h
                far_left = _overlap_start(c0 + 1, c0 + box_w - 2, far_w, w_grid, rng)
                if far_top < 0 or far_left is None:
                    continue
        elif opening == "down":
            bottom_need = food_h + near_gap + (far_h + far_gap if secondary_shape is not None else 0)
            if box_h + bottom_need > h_grid:
                continue
            r0 = rng.randint(0, h_grid - box_h - bottom_need)
            c0 = rng.randint(0, w_grid - box_w)
            food_top = r0 + box_h + near_gap
            food_left = _overlap_start(c0 + 1, c0 + box_w - 2, food_w, w_grid, rng)
            if food_left is None:
                continue
            far_top = None
            far_left = None
            if secondary_shape is not None:
                far_top = food_top + food_h + far_gap
                far_left = _overlap_start(c0 + 1, c0 + box_w - 2, far_w, w_grid, rng)
                if far_top + far_h > h_grid or far_left is None:
                    continue
        elif opening == "left":
            left_need = food_w + near_gap + (far_w + far_gap if secondary_shape is not None else 0)
            if left_need > w_grid - box_w:
                continue
            r0 = rng.randint(0, h_grid - box_h)
            c0 = rng.randint(left_need, w_grid - box_w)
            food_left = c0 - near_gap - food_w
            food_top = _overlap_start(r0 + 1, r0 + box_h - 2, food_h, h_grid, rng)
            if food_top is None:
                continue
            far_top = None
            far_left = None
            if secondary_shape is not None:
                far_left = food_left - far_gap - far_w
                far_top = _overlap_start(r0 + 1, r0 + box_h - 2, far_h, h_grid, rng)
                if far_left < 0 or far_top is None:
                    continue
        else:
            right_need = food_w + near_gap + (far_w + far_gap if secondary_shape is not None else 0)
            if box_w + right_need > w_grid:
                continue
            r0 = rng.randint(0, h_grid - box_h)
            c0 = rng.randint(0, w_grid - box_w - right_need)
            food_left = c0 + box_w + near_gap
            food_top = _overlap_start(r0 + 1, r0 + box_h - 2, food_h, h_grid, rng)
            if food_top is None:
                continue
            far_top = None
            far_left = None
            if secondary_shape is not None:
                far_left = food_left + food_w + far_gap
                far_top = _overlap_start(r0 + 1, r0 + box_h - 2, far_h, h_grid, rng)
                if far_left + far_w > w_grid or far_top is None:
                    continue

        container_shape = tuple(map(tuple, np.argwhere(_container_mask(box_h, box_w, opening))))
        container_cells = _shift_cells(container_shape, r0, c0)
        primary_cells = _shift_cells(primary_shape, food_top, food_left)
        parts = [container_cells, primary_cells]
        secondary_cells = None
        if secondary_shape is not None and far_top is not None and far_left is not None:
            secondary_cells = _shift_cells(secondary_shape, far_top, far_left)
            parts.append(secondary_cells)

        sr0, sc0, sr1, sc1 = _scene_bbox(parts)
        pad_r0 = max(0, sr0 - 1)
        pad_c0 = max(0, sc0 - 1)
        pad_r1 = min(h_grid - 1, sr1 + 1)
        pad_c1 = min(w_grid - 1, sc1 + 1)
        if occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
            continue

        primary_color = rng.choice(food_colors)
        _paint(grid, container_cells, container_color)
        _paint(grid, primary_cells, primary_color)
        occupied[[r for r, _ in container_cells], [c for _, c in container_cells]] = True
        occupied[[r for r, _ in primary_cells], [c for _, c in primary_cells]] = True

        if secondary_cells is not None:
            secondary_color = rng.choice(food_colors)
            _paint(grid, secondary_cells, secondary_color)
            occupied[[r for r, _ in secondary_cells], [c for _, c in secondary_cells]] = True
        else:
            secondary_color = None

        bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        return {
            "bbox": bbox,
            "opening": opening,
            "cavity": _cavity_cells(bbox, opening),
            "primary_food": {"cells": primary_cells, "color": primary_color},
            "secondary_food": None if secondary_cells is None else {"cells": secondary_cells, "color": secondary_color},
        }
    return None


def _place_extra_food(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    food_colors: list[int],
) -> bool:
    h_grid, w_grid = grid.shape
    for _ in range(MAX_EXTRA_FOOD_TRIES):
        shape = _random_food_shape(rng)
        food_h, food_w = _shape_dims(shape)
        if food_h > h_grid or food_w > w_grid:
            continue
        top = rng.randint(0, h_grid - food_h)
        left = rng.randint(0, w_grid - food_w)
        cells = _shift_cells(shape, top, left)
        r0, c0, r1, c1 = _bbox(cells)
        pad_r0 = max(0, r0 - 1)
        pad_c0 = max(0, c0 - 1)
        pad_r1 = min(h_grid - 1, r1 + 1)
        pad_c1 = min(w_grid - 1, c1 + 1)
        if occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
            continue
        _paint(grid, cells, rng.choice(food_colors))
        occupied[[r for r, _ in cells], [c for _, c in cells]] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    container_count = rng.choice((2, 3, 4))
    min_side = {2: 16, 3: 20, 4: 24}[container_count]
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    container_colors = rng.sample(list(range(1, 10)), container_count)
    food_pool = [c for c in range(1, 10) if c not in container_colors]
    food_colors = rng.sample(food_pool, rng.randint(2, min(4, len(food_pool))))

    openings = list(OPENINGS)
    rng.shuffle(openings)
    openings = openings[:container_count]

    want_secondary = rng.random() < 0.5
    secondary_index = rng.randrange(container_count) if want_secondary else -1

    scenes: list[dict[str, object]] = []
    for idx, opening in enumerate(openings):
        scene = _place_scene(
            grid=grid,
            occupied=occupied,
            rng=rng,
            opening=opening,
            container_color=container_colors[idx],
            food_colors=food_colors,
            force_secondary=(idx == secondary_index),
        )
        if scene is None:
            return None
        scenes.append(scene)

    for _ in range(rng.randint(0, 2)):
        _place_extra_food(grid, occupied, rng, food_colors)

    if want_secondary and scenes[secondary_index]["secondary_food"] is None:
        return None

    expected = grid.copy()
    for scene in scenes:
        primary = scene["primary_food"]
        for r, c in primary["cells"]:
            expected[r, c] = BG
        for r, c in scene["cavity"]:
            expected[r, c] = int(primary["color"])

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-food ARC input")
