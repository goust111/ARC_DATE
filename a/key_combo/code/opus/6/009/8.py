from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ROTATIONS_FROM_TOP = {
    "up": 0,
    "left": 1,
    "down": 2,
    "right": 3,
}

FIXED_COLLISION_TOP = np.array(
    [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dtype=int,
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _side_gap(mask: np.ndarray) -> str:
    h, w = mask.shape
    sides = []
    if h >= 3 and w >= 3:
        if any((not mask[0, c]) and mask[0, c - 1] and mask[0, c + 1] for c in range(1, w - 1)):
            sides.append("up")
        if any((not mask[h - 1, c]) and mask[h - 1, c - 1] and mask[h - 1, c + 1] for c in range(1, w - 1)):
            sides.append("down")
        if any((not mask[r, 0]) and mask[r - 1, 0] and mask[r + 1, 0] for r in range(1, h - 1)):
            sides.append("left")
        if any((not mask[r, w - 1]) and mask[r - 1, w - 1] and mask[r + 1, w - 1] for r in range(1, h - 1)):
            sides.append("right")
    if len(sides) != 1:
        raise ValueError(f"expected exactly one gap side, got {sides}")
    return sides[0]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for coords in _find_components(grid == BLUE):
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        rel = coords - np.array([r0, c0], dtype=int)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        side = _side_gap(mask)
        objects.append(
            {
                "bbox": bbox,
                "rel": rel,
                "direction_name": side,
                "direction": np.array(DIRS[side], dtype=int),
                "current": coords.copy(),
                "initial": coords.copy(),
                "active": True,
                "moved_steps": 0,
                "stop_reason": None,
            }
        )
    return objects


def _simulate(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    objects = _extract_objects(grid)
    while any(obj["active"] for obj in objects):
        staged = []
        for obj in objects:
            if obj["active"]:
                staged.append(obj["current"] + obj["direction"])
            else:
                staged.append(obj["current"])

        occupancy: dict[tuple[int, int], int] = defaultdict(int)
        for coords in staged:
            for r, c in coords:
                occupancy[(int(r), int(c))] += 1

        colliding = []
        for obj, coords in zip(objects, staged):
            if not obj["active"]:
                colliding.append(False)
                continue
            oob = bool(
                (coords[:, 0] < 0).any()
                or (coords[:, 0] >= h).any()
                or (coords[:, 1] < 0).any()
                or (coords[:, 1] >= w).any()
            )
            overlap = any(occupancy[(int(r), int(c))] > 1 for r, c in coords)
            colliding.append(oob or overlap)

        for obj, coords, hit in zip(objects, staged, colliding):
            if not obj["active"]:
                continue
            if hit:
                obj["active"] = False
                obj["stop_reason"] = "collision"
            else:
                obj["current"] = coords
                obj["moved_steps"] += 1
    for obj in objects:
        if obj["stop_reason"] is None:
            obj["stop_reason"] = "boundary"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each blue foreground object in the direction encoded by its bbox gap, then recolor by whether horizontal or vertical motion actually happened.

    Input structure:
    - The grid is a yellow background with 4 to 6 disconnected blue foreground objects.
    - Every object is 4-connected, has area 4 to 8, and spans a bbox of at least 3x3.
    - Exactly one side of each object's bbox has a single border gap: a background cell on that bbox edge with object pixels on both sides along the same edge.

    Rule:
    - Detect the unique gap side of each object's bbox: top, bottom, left, or right.
    - That side determines the object's motion direction: top -> up, bottom -> down, left -> left, right -> right.
    - All objects move synchronously one cell per step along their own directions.
    - After each step, any moving object whose tentative position goes out of bounds or overlaps any other object's tentative/current position reverts that step and stops; the other non-colliding objects keep the step.
    - When all objects have stopped, recolor moved horizontal objects red, moved vertical objects green, and objects that never moved blue.

    Key insight:
    - The bbox gap is not a decorative notch: it selects a personal slide direction, and synchronous collision handling can change whether an object moves at all, which in turn determines its final color.
    """
    arr = np.asarray(grid, dtype=int)
    out = np.full_like(arr, YELLOW)
    for obj in _simulate(arr):
        color = BLUE
        if obj["moved_steps"] > 0:
            color = RED if obj["direction"][1] != 0 else GREEN
        coords = obj["current"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _rotate_to_direction(mask: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(mask, ROTATIONS_FROM_TOP[direction])


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int = BLUE) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask > 0)
    grid[top + coords[:, 0], left + coords[:, 1]] = color
    h, w = mask.shape
    return top, left, top + h - 1, left + w - 1


def _connected(mask: np.ndarray) -> bool:
    comps = _find_components(mask.astype(bool))
    return len(comps) == 1


def _build_top_gap_catalog() -> list[np.ndarray]:
    catalog = []
    for h, w in ((3, 3), (3, 4), (4, 3)):
        total = h * w
        for bits in range(1 << total):
            area = bits.bit_count()
            if area < 4 or area > 8:
                continue
            mask = np.zeros((h, w), dtype=int)
            for i in range(total):
                if bits & (1 << i):
                    mask[i // w, i % w] = 1
            if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                continue
            coords = np.argwhere(mask > 0)
            if tuple(_bbox_from_coords(coords)) != (0, 0, h - 1, w - 1):
                continue
            if not _connected(mask):
                continue
            try:
                if _side_gap(mask.astype(bool)) == "up":
                    catalog.append(mask)
            except ValueError:
                pass
    random.shuffle(catalog)
    return catalog


TOP_GAP_CATALOG = _build_top_gap_catalog()


def _random_object_mask(direction: str) -> np.ndarray:
    mask = random.choice(TOP_GAP_CATALOG)
    return _rotate_to_direction(mask, direction)


def _random_free_placement(
    occupied_boxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    mask: np.ndarray,
    row_choices: list[int] | None = None,
    col_choices: list[int] | None = None,
    tries: int = 300,
) -> tuple[int, int] | None:
    mh, mw = mask.shape
    if row_choices is None:
        row_choices = list(range(0, h - mh + 1))
    if col_choices is None:
        col_choices = list(range(0, w - mw + 1))
    if not row_choices or not col_choices:
        return None
    for _ in range(tries):
        top = random.choice(row_choices)
        left = random.choice(col_choices)
        box = (top, left, top + mh - 1, left + mw - 1)
        if any(_boxes_too_close(box, other) for other in occupied_boxes):
            continue
        return top, left
    return None


def _allowed_rows(grid_h: int, mask_h: int, forbidden: tuple[int, int]) -> list[int]:
    f0, f1 = forbidden
    rows = []
    for top in range(0, grid_h - mask_h + 1):
        if top + mask_h - 1 < f0 or top > f1:
            rows.append(top)
    return rows


def _allowed_cols(grid_w: int, mask_w: int, forbidden: tuple[int, int]) -> list[int]:
    f0, f1 = forbidden
    cols = []
    for left in range(0, grid_w - mask_w + 1):
        if left + mask_w - 1 < f0 or left > f1:
            cols.append(left)
    return cols


def _validate_generated_grid(grid: np.ndarray, expect_immediate_block: bool) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    objs = _extract_objects(grid)
    if not (4 <= len(objs) <= 6):
        return False
    boxes = [obj["bbox"] for obj in objs]
    for i, box_i in enumerate(boxes):
        r0, c0, r1, c1 = box_i
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            return False
        area = len(objs[i]["rel"])
        if not (4 <= area <= 8):
            return False
        for j in range(i + 1, len(boxes)):
            if _boxes_too_close(box_i, boxes[j]):
                return False

    sim = _simulate(grid)
    moved_horizontal = any(obj["moved_steps"] > 0 and obj["direction"][1] != 0 for obj in sim)
    moved_vertical = any(obj["moved_steps"] > 0 and obj["direction"][0] != 0 for obj in sim)
    collision_stop = any(obj["stop_reason"] == "collision" for obj in sim)
    still_blue = any(obj["moved_steps"] == 0 for obj in sim)
    moved_after_collision = any(obj["stop_reason"] == "collision" and obj["moved_steps"] > 0 for obj in sim)
    if not moved_horizontal or not moved_vertical or not collision_stop:
        return False
    if expect_immediate_block and not still_blue:
        return False
    if (not expect_immediate_block) and not moved_after_collision:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    colors = set(int(v) for v in np.unique(out))
    if not ({RED, GREEN} <= colors):
        return False
    if expect_immediate_block and BLUE not in colors:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(16, 22)
    grid_w = random.randint(16, 22)
    grid = np.full((grid_h, grid_w), YELLOW, dtype=int)
    occupied_boxes: list[tuple[int, int, int, int]] = []

    pair_axis = random.choice(["horizontal", "vertical"])
    immediate_block = random.random() < 0.6
    pair_gap = 1 if immediate_block else random.randint(2, 4)
    pair_mask_a = _rotate_to_direction(FIXED_COLLISION_TOP, "right" if pair_axis == "horizontal" else "down")
    pair_mask_b = _rotate_to_direction(FIXED_COLLISION_TOP, "left" if pair_axis == "horizontal" else "up")
    ah, aw = pair_mask_a.shape
    bh, bw = pair_mask_b.shape

    if pair_axis == "horizontal":
        total_w = aw + pair_gap + bw
        if total_w > grid_w - 2 or ah > grid_h:
            return None
        top = random.randint(0, grid_h - ah)
        left = random.randint(1, grid_w - total_w - 1)
        box_a = _place_mask(grid, pair_mask_a, top, left)
        box_b = _place_mask(grid, pair_mask_b, top, left + aw + pair_gap)
    else:
        total_h = ah + pair_gap + bh
        if total_h > grid_h - 2 or aw > grid_w:
            return None
        top = random.randint(1, grid_h - total_h - 1)
        left = random.randint(0, grid_w - aw)
        box_a = _place_mask(grid, pair_mask_a, top, left)
        box_b = _place_mask(grid, pair_mask_b, top + ah + pair_gap, left)
    occupied_boxes.extend([box_a, box_b])

    free_h_dir = random.choice(["left", "right"])
    free_h_mask = _random_object_mask(free_h_dir)
    hmask_h, hmask_w = free_h_mask.shape
    pair_rows = (min(box_a[0], box_b[0]), max(box_a[2], box_b[2]))
    row_choices = _allowed_rows(grid_h, hmask_h, pair_rows)
    col_choices = list(range(0, grid_w - hmask_w + 1))
    pos = _random_free_placement(occupied_boxes, grid_h, grid_w, free_h_mask, row_choices=row_choices, col_choices=col_choices)
    if pos is None:
        return None
    occupied_boxes.append(_place_mask(grid, free_h_mask, pos[0], pos[1]))

    free_v_dir = random.choice(["up", "down"])
    free_v_mask = _random_object_mask(free_v_dir)
    vmask_h, vmask_w = free_v_mask.shape
    pair_cols = (min(box_a[1], box_b[1]), max(box_a[3], box_b[3]))
    row_choices = list(range(0, grid_h - vmask_h + 1))
    col_choices = _allowed_cols(grid_w, vmask_w, pair_cols)
    pos = _random_free_placement(occupied_boxes, grid_h, grid_w, free_v_mask, row_choices=row_choices, col_choices=col_choices)
    if pos is None:
        return None
    occupied_boxes.append(_place_mask(grid, free_v_mask, pos[0], pos[1]))

    extras = random.randint(0, 2)
    for _ in range(extras):
        direction = random.choice(list(DIRS))
        mask = _random_object_mask(direction)
        pos = _random_free_placement(occupied_boxes, grid_h, grid_w, mask)
        if pos is None:
            return None
        occupied_boxes.append(_place_mask(grid, mask, pos[0], pos[1]))

    if not _validate_generated_grid(grid, expect_immediate_block=immediate_block):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
