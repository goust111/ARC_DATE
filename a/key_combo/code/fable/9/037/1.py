from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
NON_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 500


def _corner_offset(direction: tuple[int, int], size: int) -> tuple[int, int]:
    dr, dc = direction
    return (0 if dr < 0 else size - 1), (0 if dc < 0 else size - 1)


def _notch_coord(top: int, left: int, size: int, direction: tuple[int, int]) -> tuple[int, int]:
    ro, co = _corner_offset(direction, size)
    return top + ro, left + co


def _hit_corner_for_ray(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dr, -dc


def _top_left_from_hit_corner(
    hit_r: int, hit_c: int, size: int, ray_direction: tuple[int, int]
) -> tuple[int, int]:
    ro, co = _corner_offset(_hit_corner_for_ray(ray_direction), size)
    return hit_r - ro, hit_c - co


def _bbox_contains(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _occupied_cells(obj: dict) -> list[tuple[int, int]]:
    top, left = obj["top"], obj["left"]
    size = obj["size"]
    notch = obj["notch"]
    cells: list[tuple[int, int]] = []
    for r in range(top, top + size):
        for c in range(left, left + size):
            if (r, c) != notch:
                cells.append((r, c))
    return cells


def _paint_object(grid: np.ndarray, obj: dict) -> None:
    for r, c in _occupied_cells(obj):
        grid[r, c] = obj["color"]


def _component_dicts(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _detect_objects(grid: np.ndarray) -> list[dict]:
    objs: list[dict] = []
    for comp in _component_dicts(grid):
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != width or height not in (2, 3):
            continue
        size = height
        if comp["area"] != size * size - 1:
            continue
        color = int(comp["color"])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((patch != 0) & (patch != color)):
            continue
        zeros = [(r0 + rr, c0 + cc) for rr in range(size) for cc in range(size) if int(patch[rr, cc]) == 0]
        if len(zeros) != 1:
            continue
        notch = zeros[0]
        corner_to_dir = {
            (r0, c0): (-1, -1),
            (r0, c1): (-1, 1),
            (r1, c0): (1, -1),
            (r1, c1): (1, 1),
        }
        if notch not in corner_to_dir:
            continue
        objs.append(
            {
                "color": color,
                "top": r0,
                "left": c0,
                "size": size,
                "bbox": (r0, c0, r1, c1),
                "notch": notch,
                "direction": corner_to_dir[notch],
            }
        )
    assert objs, "expected at least one notched square"
    big = [obj for obj in objs if obj["size"] == 3]
    assert len(big) == 1, "expected exactly one 3x3 notched square"
    assert all(obj["size"] in (2, 3) for obj in objs)
    return objs


def _trace(
    objects: list[dict], shape: tuple[int, int]
) -> tuple[np.ndarray, list[int], list[list[tuple[int, int]]], list[int | None]]:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for obj in objects:
        _paint_object(out, obj)

    bbox_owner = np.full((h, w), -1, dtype=int)
    for idx, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["bbox"]
        bbox_owner[r0 : r1 + 1, c0 : c1 + 1] = idx

    starters = [idx for idx, obj in enumerate(objects) if obj["size"] == 3]
    assert len(starters) == 1
    start_id = starters[0]

    activated: list[int] = []
    visited: set[int] = set()
    queue: deque[int] = deque([start_id])
    ray_cells: list[list[tuple[int, int]]] = [[] for _ in objects]
    hits: list[int | None] = [None for _ in objects]

    while queue:
        idx = queue.popleft()
        if idx in visited:
            continue
        visited.add(idx)
        activated.append(idx)
        obj = objects[idx]
        color = int(obj["color"])
        dr, dc = obj["direction"]
        r, c = obj["notch"]
        emitted: list[tuple[int, int]] = []
        while 0 <= r < h and 0 <= c < w:
            out[r, c] = color
            emitted.append((r, c))
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            hit_id = int(bbox_owner[nr, nc])
            if hit_id != -1 and hit_id != idx:
                hits[idx] = hit_id
                if hit_id not in visited:
                    queue.append(hit_id)
                break
            r, c = nr, nc
        ray_cells[idx] = emitted

    return out, activated, ray_cells, hits


def _grid_from_objects(shape: tuple[int, int], objects: Iterable[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        _paint_object(grid, obj)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite the unique large notched square and let only the hit squares relay diagonal rays.

    Input structure:
    - The grid has a black background.
    - It contains several monochrome 2x2 squares missing one corner and exactly one 3x3 square missing one corner.
    - Each missing corner is one of the four bbox corners, so it visibly encodes one diagonal direction.

    Rule:
    - Keep every notched square in place.
    - Start from the unique 3x3 square: fill its missing corner and extend a same-color diagonal ray outward from that corner.
    - A ray stops when its next diagonal step would leave the grid or enter the bbox of another notched square; that first reached square becomes activated and emits its own ray once.
    - Already activated squares do not emit again, and never-hit squares remain unchanged.

    Key insight:
    - The missing corner tells you where each ray starts and which diagonal it follows, but reachability is causal: only squares that are physically struck by an earlier ray get to fire.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _detect_objects(g)
    out, _activated, _ray_cells, _hits = _trace(objects, g.shape)
    return out


def _path_is_clear(source: dict, target_bbox: tuple[int, int, int, int], placed: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    dr, dc = source["direction"]
    r, c = source["notch"]
    target_idx = len(placed)
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        if _bbox_contains(target_bbox, nr, nc):
            return True
        for obj in placed:
            if _bbox_contains(obj["bbox"], nr, nc):
                return False
        r, c = nr, nc
        if len(placed) > target_idx + h + w:
            return False


def _candidate_target(
    source: dict,
    size: int,
    notch_dir: tuple[int, int],
    shape: tuple[int, int],
    placed: list[dict],
) -> dict | None:
    h, w = shape
    gaps = list(range(3, max(h, w)))
    random.shuffle(gaps)
    for gap in gaps:
        hit_r = source["notch"][0] + gap * source["direction"][0]
        hit_c = source["notch"][1] + gap * source["direction"][1]
        top, left = _top_left_from_hit_corner(hit_r, hit_c, size, source["direction"])
        if top < 0 or left < 0 or top + size > h or left + size > w:
            continue
        bbox = (top, left, top + size - 1, left + size - 1)
        if any(not _boxes_separate(bbox, obj["bbox"], pad=1) for obj in placed):
            continue
        obj = {
            "color": -1,
            "top": top,
            "left": left,
            "size": size,
            "bbox": bbox,
            "notch": _notch_coord(top, left, size, notch_dir),
            "direction": notch_dir,
        }
        if _path_is_clear(source, bbox, placed, shape):
            return obj
    return None


def _ray_targets(obj_idx: int, objects: list[dict], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    dr, dc = objects[obj_idx]["direction"]
    r, c = objects[obj_idx]["notch"]
    hits: list[tuple[int, int]] = []
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        for target_idx, other in enumerate(objects):
            if target_idx != obj_idx and _bbox_contains(other["bbox"], nr, nc):
                hits.append((abs(nr - r) + abs(nc - c), target_idx))
                break
        r, c = nr, nc
    hits.sort()
    ordered: list[int] = []
    seen: set[int] = set()
    for _dist, target_idx in hits:
        if target_idx not in seen:
            seen.add(target_idx)
            ordered.append(target_idx)
    return ordered


def _choose_loop_direction(last: dict, earlier_ids: list[int], objects: list[dict], shape: tuple[int, int]) -> tuple[int, int] | None:
    current_idx = len(objects) - 1
    for direction in random.sample(DIRS, len(DIRS)):
        trial = dict(last)
        trial["direction"] = direction
        trial["notch"] = _notch_coord(trial["top"], trial["left"], trial["size"], direction)
        trial_objects = objects[:-1] + [trial]
        targets = _ray_targets(current_idx, trial_objects, shape)
        if targets and targets[0] in earlier_ids:
            return direction
    return None


def _choose_terminal_direction(last: dict, objects: list[dict], shape: tuple[int, int]) -> tuple[int, int] | None:
    current_idx = len(objects) - 1
    options = random.sample(DIRS, len(DIRS))
    fallback: tuple[int, int] | None = None
    for direction in options:
        trial = dict(last)
        trial["direction"] = direction
        trial["notch"] = _notch_coord(trial["top"], trial["left"], trial["size"], direction)
        trial_objects = objects[:-1] + [trial]
        targets = _ray_targets(current_idx, trial_objects, shape)
        if not targets:
            return direction
        if fallback is None:
            fallback = direction
    return fallback


def _unique_ray_contributions(objects: list[dict], shape: tuple[int, int]) -> bool:
    out, activated, ray_cells, _hits = _trace(objects, shape)
    painted = _grid_from_objects(shape, objects) != 0
    seen = painted.copy()
    for idx in activated:
        new_cells = [(r, c) for r, c in ray_cells[idx] if not seen[r, c]]
        if not new_cells:
            return False
        for r, c in ray_cells[idx]:
            seen[r, c] = True
    return np.any(out != _grid_from_objects(shape, objects))


def _valid_example(objects: list[dict], shape: tuple[int, int]) -> bool:
    inp = _grid_from_objects(shape, objects)
    out, activated, ray_cells, hits = _trace(objects, shape)
    if not np.array_equal(main(inp), out):
        return False
    if len([obj for obj in objects if obj["size"] == 3]) != 1:
        return False
    if len(activated) < 3 or len(activated) >= len(objects):
        return False
    dirs = {objects[idx]["direction"] for idx in activated}
    if len(dirs) < 2:
        return False
    if not any(hits[idx] is None for idx in activated):
        return False
    if not _unique_ray_contributions(objects, shape):
        return False
    first = activated[0]
    if hits[first] is None:
        return False
    later = activated[2:]
    if not later:
        return False
    first_ray_cells = set(ray_cells[first])
    for idx in later:
        if objects[idx]["notch"] in first_ray_cells:
            return False
    return True


def _place_distractors(shape: tuple[int, int], objects: list[dict], distractor_count: int) -> list[dict] | None:
    base_out, base_activated, _base_rays, _base_hits = _trace(objects, shape)
    occupied = [dict(obj) for obj in objects]
    tries = 0
    while len(occupied) < len(objects) + distractor_count and tries < 200:
        tries += 1
        size = 2
        direction = random.choice(DIRS)
        h, w = shape
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        bbox = (top, left, top + size - 1, left + size - 1)
        if any(not _boxes_separate(bbox, obj["bbox"], pad=1) for obj in occupied):
            continue
        obj = {
            "color": -1,
            "top": top,
            "left": left,
            "size": size,
            "bbox": bbox,
            "notch": _notch_coord(top, left, size, direction),
            "direction": direction,
        }
        trial = occupied + [obj]
        temp_out, temp_activated, _temp_rays, _temp_hits = _trace(trial, shape)
        mask = np.zeros(shape, dtype=bool)
        for r, c in _occupied_cells(obj):
            mask[r, c] = True
        if temp_activated != base_activated:
            continue
        if not np.array_equal(temp_out[~mask], base_out[~mask]):
            continue
        occupied.append(obj)
    return occupied if len(occupied) == len(objects) + distractor_count else None


def _assign_colors(objects: list[dict]) -> None:
    colors = random.sample(NON_BLACK, len(objects))
    for obj, color in zip(objects, colors):
        obj["color"] = color


def _generate_active_layout(shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    active_smalls = random.randint(2, 4)
    want_loop = active_smalls >= 3 and random.random() < 0.35

    size = 3
    for _ in range(120):
        big_dir = random.choice(DIRS)
        top = random.randint(1, h - size - 2)
        left = random.randint(1, w - size - 2)
        big = {
            "color": -1,
            "top": top,
            "left": left,
            "size": size,
            "bbox": (top, left, top + size - 1, left + size - 1),
            "notch": _notch_coord(top, left, size, big_dir),
            "direction": big_dir,
        }
        objects = [big]
        ok = True
        for step in range(active_smalls):
            source = objects[-1]
            dirs = random.sample(DIRS, len(DIRS))
            placed = None
            for notch_dir in dirs:
                candidate = _candidate_target(source, 2, notch_dir, shape, objects)
                if candidate is not None:
                    placed = candidate
                    break
            if placed is None:
                ok = False
                break
            objects.append(placed)
        if not ok:
            continue

        if want_loop:
            loop_dir = _choose_loop_direction(objects[-1], list(range(len(objects) - 1)), objects, shape)
            if loop_dir is None:
                continue
            objects[-1]["direction"] = loop_dir
            objects[-1]["notch"] = _notch_coord(
                objects[-1]["top"], objects[-1]["left"], objects[-1]["size"], loop_dir
            )
        else:
            direction = _choose_terminal_direction(objects[-1], objects, shape)
            if direction is None:
                continue
            objects[-1]["direction"] = direction
            objects[-1]["notch"] = _notch_coord(
                objects[-1]["top"], objects[-1]["left"], objects[-1]["size"], direction
            )

        return objects
    return None


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(14, 24), random.randint(14, 24))
    active_objects = _generate_active_layout(shape)
    if active_objects is None:
        return None

    distractors = random.randint(1, 2)
    objects = _place_distractors(shape, active_objects, distractors)
    if objects is None:
        return None
    _assign_colors(objects)

    if not _valid_example(objects, shape):
        return None
    return _grid_from_objects(shape, objects)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notched-chain sample")
