from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


PALETTE = [1, 2, 3, 4, 5, 6, 7]
GRID_SIDE_RANGE = (17, 22)
OBJECT_COUNT_RANGE = (5, 7)
AREA_CHOICES = tuple(range(4, 16))
GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 60
CENTER_MARGIN = 1
MIN_CENTER_GAP = 3


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_objects(grid: np.ndarray, bg: int = 0) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            bbox = _bbox_from_coords(coords_arr)
            objects.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": bbox,
                    "area": int(coords_arr.shape[0]),
                }
            )
    return objects


def _center_col(bbox: tuple[int, int, int, int]) -> int:
    return (int(bbox[1]) + int(bbox[3])) // 2


def _select_survivors(objects: list[dict]) -> list[dict]:
    by_center: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        by_center[_center_col(obj["bbox"])].append(obj)
    return [max(group, key=lambda obj: int(obj["area"])) for group in by_center.values()]


def _rings_from_colors(colors: list[int]) -> np.ndarray:
    if not colors:
        return np.zeros((0, 0), dtype=int)
    side = 2 * len(colors) - 1
    out = np.zeros((side, side), dtype=int)
    for layer, color in enumerate(colors):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter objects by shared projection columns and redraw the survivors as concentric square rings.

    Input structure:
    - The grid has a black background and several separated monochromatic 4-connected objects.
    - Each object is identified by its bounding box, area, and color.
    - Multiple objects may share the same bounding-box center column while the last row stays empty.

    Rule:
    - Detect every foreground object and compute the horizontal center column of its bbox.
    - For each shared center column, keep only the largest-area object and discard the smaller ones.
    - Let N be the number of surviving objects; create a black square of side 2N-1.
    - Sort the survivors by decreasing area and paint their colors as concentric square rings from outside to inside.

    Key insight:
    - The vertical projection column is not drawn directly; it only decides which objects survive, and the same area ranking then determines the ring order.
    """
    g = np.asarray(grid, dtype=int)
    objects = _find_objects(g, bg=0)
    survivors = _select_survivors(objects)
    ordered = sorted(survivors, key=lambda obj: int(obj["area"]), reverse=True)
    return _rings_from_colors([int(obj["color"]) for obj in ordered])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_blob_mask(area: int) -> np.ndarray:
    area = int(area)
    feasible = [
        (h, w)
        for h in range(2, 6)
        for w in range(2, 6)
        if area <= h * w and area < h * w and h * w <= area + 6
    ]
    if not feasible:
        feasible = [(h, w) for h in range(2, 6) for w in range(2, 6) if area <= h * w and area < h * w]

    for _ in range(SHAPE_RETRIES):
        h, w = random.choice(feasible)
        cells = {(random.randrange(h), random.randrange(w))}
        frontier = set()

        def add_frontier(r: int, c: int) -> None:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(*next(iter(cells)))
        while len(cells) < area:
            pick = random.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            add_frontier(*pick)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) != area:
            continue
        if area == mask.shape[0] * mask.shape[1]:
            continue
        if random.choice([False, True]):
            mask = np.flipud(mask)
        if random.choice([False, True]):
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        return np.asarray(mask, dtype=bool)

    raise RuntimeError(f"failed to build irregular blob of area {area}")


def _partition_group_sizes(n_objects: int) -> list[int]:
    n_groups = random.randint(3, n_objects - 1)
    sizes = [1] * n_groups
    for _ in range(n_objects - n_groups):
        sizes[random.randrange(n_groups)] += 1
    random.shuffle(sizes)
    return sizes


def _choose_group_centers(groups: list[list[dict]], width: int) -> list[int] | None:
    intervals = []
    for group in groups:
        min_center = 0
        max_center = width - 1
        for obj in group:
            patch_w = int(obj["mask"].shape[1])
            left = (patch_w - 1) // 2
            right = patch_w // 2
            min_center = max(min_center, CENTER_MARGIN + left)
            max_center = min(max_center, width - 1 - CENTER_MARGIN - right)
        if min_center > max_center:
            return None
        candidates = list(range(min_center, max_center + 1))
        random.shuffle(candidates)
        intervals.append(candidates)

    order = sorted(range(len(groups)), key=lambda i: len(intervals[i]))
    chosen: dict[int, int] = {}

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for center in intervals[idx]:
            if all(abs(center - chosen[other]) >= MIN_CENTER_GAP for other in chosen):
                chosen[idx] = center
                if search(pos + 1):
                    return True
                del chosen[idx]
        return False

    if not search(0):
        return None
    return [chosen[i] for i in range(len(groups))]


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    patch_h, patch_w = mask.shape
    if top < 0 or left < 0 or top + patch_h > h - 1 or left + patch_w > w:
        return False

    for r, c in np.argwhere(mask):
        rr = int(top + r)
        cc = int(left + c)
        if occupied[rr, cc]:
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _paint_patch(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    for r, c in np.argwhere(mask):
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(color)
        occupied[rr, cc] = True


def _place_objects(height: int, width: int, objects: list[dict], groups: list[list[dict]], centers: list[int]) -> np.ndarray | None:
    for _ in range(PLACEMENT_RETRIES):
        grid = np.zeros((height, width), dtype=int)
        occupied = np.zeros((height, width), dtype=bool)
        placed = True

        for obj in sorted(objects, key=lambda item: int(item["area"]), reverse=True):
            center = int(centers[int(obj["group_id"])])
            patch_w = int(obj["mask"].shape[1])
            left = center - (patch_w - 1) // 2
            rows = list(range(0, height - int(obj["mask"].shape[0])))
            random.shuffle(rows)

            found = False
            for top in rows:
                if _can_place(occupied, obj["mask"], top, left):
                    _paint_patch(grid, occupied, obj["mask"], int(obj["color"]), top, left)
                    found = True
                    break
            if not found:
                placed = False
                break

        if placed and np.all(grid[-1] == 0):
            return grid
    return None


def _build_objects() -> tuple[list[dict], list[list[dict]]]:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, n_objects)
    areas = random.sample(AREA_CHOICES, n_objects)

    objects = []
    for color, area in zip(colors, areas):
        objects.append(
            {
                "color": int(color),
                "area": int(area),
                "mask": _random_blob_mask(int(area)),
            }
        )

    random.shuffle(objects)
    group_sizes = _partition_group_sizes(n_objects)
    groups: list[list[dict]] = []
    cursor = 0
    for gid, size in enumerate(group_sizes):
        group = []
        for obj in objects[cursor : cursor + size]:
            obj["group_id"] = gid
            group.append(obj)
        groups.append(group)
        cursor += size
    return objects, groups


def _oracle_valid(grid: np.ndarray, expected_count: int) -> bool:
    objects = _find_objects(grid, bg=0)
    if len(objects) != expected_count:
        return False
    if len({int(obj["color"]) for obj in objects}) != expected_count:
        return False
    areas = [int(obj["area"]) for obj in objects]
    if len(set(areas)) != len(areas):
        return False
    centers = [_center_col(obj["bbox"]) for obj in objects]
    if len(set(centers)) == len(centers):
        return False
    survivors = _select_survivors(objects)
    if len(survivors) < 3:
        return False

    expected = _rings_from_colors(
        [int(obj["color"]) for obj in sorted(survivors, key=lambda obj: int(obj["area"]), reverse=True)]
    )
    actual = main(grid)
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(*GRID_SIDE_RANGE)
        width = random.randint(*GRID_SIDE_RANGE)

        try:
            objects, groups = _build_objects()
        except RuntimeError:
            continue

        centers = _choose_group_centers(groups, width)
        if centers is None:
            continue

        grid = _place_objects(height, width, objects, groups, centers)
        if grid is None:
            continue
        if _oracle_valid(grid, len(objects)):
            return grid

    raise RuntimeError("failed to generate a valid input")
