from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIR_BY_CENTER = {
    RED: (1, 0),      # down
    BLUE: (0, -1),    # left
    GREEN: (0, 1),    # right
    YELLOW: (-1, 0),  # up
}

DIR_NAME = {
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
    (-1, 0): "up",
}

BASE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
)

GEN_HEIGHT_RANGE = (18, 22)
GEN_WIDTH_RANGE = (18, 22)
GEN_FRAME_COUNT_RANGE = (3, 4)
GEN_OBJECT_COUNT_RANGE = (3, 5)
GEN_DISTRACTOR_COUNT_RANGE = (1, 4)
GEN_FRAME_LOCAL_DISTANCE = 9
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _frame_ring(center_r: int, center_c: int) -> list[tuple[int, int]]:
    return [
        (center_r - 1, center_c - 1),
        (center_r - 1, center_c),
        (center_r - 1, center_c + 1),
        (center_r, center_c - 1),
        (center_r, center_c + 1),
        (center_r + 1, center_c - 1),
        (center_r + 1, center_c),
        (center_r + 1, center_c + 1),
    ]


def _find_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    frames = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            if center not in DIR_BY_CENTER:
                continue
            if all(int(grid[rr, cc]) == GREY for rr, cc in _frame_ring(r, c)):
                frames.append(
                    {
                        "center": (r, c),
                        "center_color": center,
                        "direction": DIR_BY_CENTER[center],
                    }
                )
    return frames


def _find_orange_objects(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != ORANGE:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == ORANGE:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(sorted(coords), dtype=int)
            objects.append(arr)
    objects.sort(key=lambda coords: _bbox(coords))
    return objects


def _centroid_distance_num(coords: np.ndarray, center: tuple[int, int]) -> int:
    area = int(coords.shape[0])
    sum_r = int(coords[:, 0].sum())
    sum_c = int(coords[:, 1].sum())
    r, c = center
    return abs(sum_r - area * int(r)) + abs(sum_c - area * int(c))


def _assign_frame(coords: np.ndarray, frames: list[dict]) -> tuple[int, list[int]]:
    dist_nums = [_centroid_distance_num(coords, frame["center"]) for frame in frames]
    best = min(dist_nums)
    winners = [i for i, value in enumerate(dist_nums) if value == best]
    if len(winners) != 1:
        raise AssertionError("each orange object must have a unique nearest frame")
    return winners[0], dist_nums


def _assign_frame_or_none(coords: np.ndarray, frames: list[dict]) -> tuple[int, list[int]] | None:
    dist_nums = [_centroid_distance_num(coords, frame["center"]) for frame in frames]
    best = min(dist_nums)
    winners = [i for i, value in enumerate(dist_nums) if value == best]
    if len(winners) != 1:
        return None
    return winners[0], dist_nums


def _boundary_distance(coords: np.ndarray, direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    min_r = int(coords[:, 0].min())
    max_r = int(coords[:, 0].max())
    min_c = int(coords[:, 1].min())
    max_c = int(coords[:, 1].max())
    if direction == (0, 1):
        return w - 1 - max_c
    if direction == (0, -1):
        return min_c
    if direction == (1, 0):
        return h - 1 - max_r
    if direction == (-1, 0):
        return min_r
    raise ValueError(f"unknown direction: {direction}")


def _dominant_side(coords: np.ndarray, center: tuple[int, int]) -> str:
    mean_r = float(coords[:, 0].mean())
    mean_c = float(coords[:, 1].mean())
    dr = mean_r - float(center[0])
    dc = mean_c - float(center[1])
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _nearest_wall_dirs(coords: np.ndarray, shape: tuple[int, int]) -> set[str]:
    min_r = int(coords[:, 0].min())
    max_r = int(coords[:, 0].max())
    min_c = int(coords[:, 1].min())
    max_c = int(coords[:, 1].max())
    h, w = shape
    dists = {
        "up": min_r,
        "down": h - 1 - max_r,
        "left": min_c,
        "right": w - 1 - max_c,
    }
    best = min(dists.values())
    return {name for name, value in dists.items() if value == best}


def _head_on(a: dict, b: dict) -> bool:
    dir_a = a["direction"]
    dir_b = b["direction"]
    coords_a = a["coords"]
    coords_b = b["coords"]
    rows_a = set(map(int, coords_a[:, 0]))
    rows_b = set(map(int, coords_b[:, 0]))
    cols_a = set(map(int, coords_a[:, 1]))
    cols_b = set(map(int, coords_b[:, 1]))
    box_a = _bbox(coords_a)
    box_b = _bbox(coords_b)

    if {dir_a, dir_b} == {(0, 1), (0, -1)} and rows_a & rows_b:
        return box_a[3] < box_b[1] or box_b[3] < box_a[1]
    if {dir_a, dir_b} == {(1, 0), (-1, 0)} and cols_a & cols_b:
        return box_a[2] < box_b[0] or box_b[2] < box_a[0]
    return False


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert 3 <= len(frames) <= 4
    objects = _find_orange_objects(g)
    assert 3 <= len(objects) <= 5

    entries = []
    for index, coords in enumerate(objects):
        frame_idx, dist_nums = _assign_frame(coords, frames)
        frame = frames[frame_idx]
        direction = frame["direction"]
        bound_dist = _boundary_distance(coords, direction, g.shape)
        entries.append(
            {
                "id": index,
                "coords": coords,
                "bbox": _bbox(coords),
                "frame_idx": frame_idx,
                "frame_center": frame["center"],
                "direction": direction,
                "direction_name": DIR_NAME[direction],
                "frame_dist_nums": dist_nums,
                "boundary_distance": bound_dist,
                "dominant_side": _dominant_side(coords, frame["center"]),
                "nearest_walls": _nearest_wall_dirs(coords, g.shape),
            }
        )

    entries.sort(
        key=lambda item: (
            item["boundary_distance"],
            item["bbox"][0],
            item["bbox"][1],
            item["bbox"][2],
            item["bbox"][3],
        )
    )

    h, w = g.shape
    out = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    results = []
    for item in entries:
        coords = item["coords"]
        dr, dc = item["direction"]
        shift = 0
        stop_reason = "wall"

        while True:
            moved = coords + np.array([[dr, dc]], dtype=int) * (shift + 1)
            rows = moved[:, 0]
            cols = moved[:, 1]
            out_of_bounds = bool(
                np.any(rows < 0) or np.any(rows >= h) or np.any(cols < 0) or np.any(cols >= w)
            )
            collision = False
            if not out_of_bounds:
                collision = bool(np.any(occupied[rows, cols]))
            if out_of_bounds or collision:
                stop_reason = "object" if collision and not out_of_bounds else "wall"
                break
            shift += 1

        final_coords = coords + np.array([[dr, dc]], dtype=int) * shift
        out[final_coords[:, 0], final_coords[:, 1]] = ORANGE
        occupied[final_coords[:, 0], final_coords[:, 1]] = True

        result = dict(item)
        result["shift"] = shift
        result["stop_reason"] = stop_reason
        result["final_coords"] = final_coords
        results.append(result)

    return out, frames, results


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each orange object in the direction encoded by the center color of its nearest grey 3x3 frame.

    Input structure:
    - The background is black.
    - There are 3x3 grey frames whose eight border cells are grey and whose centers are one of blue, red, green, or yellow.
    - Orange 4-connected objects are scattered separately, and extra isolated grey pixels may appear as distractors.

    Rule:
    - Detect every complete 3x3 grey frame and decode its center color as a direction: red=down, blue=left, green=right, yellow=up.
    - For each orange object, compare the Manhattan distance from its centroid to every frame center and assign the object to its unique nearest frame.
    - Process objects from smallest distance-to-target-boundary to largest, sliding each object as a rigid shape until its next step would leave the grid or hit an already settled orange object.
    - Erase all frames and grey distractors; keep only the final orange object positions on black.

    Key insight:
    - The frame centers are not output colors: they are hidden motion commands, and each object must first be matched to its nearest valid frame before the gravity-like sliding order can be applied.
    """
    out, _frames, _results = _simulate(np.asarray(grid, dtype=int))
    return out


def _normalize_shape(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(list(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    arr = np.unique(arr, axis=0)
    return arr[np.lexsort((arr[:, 1], arr[:, 0]))]


def _random_shape() -> np.ndarray:
    coords = np.asarray(BASE_SHAPES[random.randrange(len(BASE_SHAPES))], dtype=int).copy()
    if random.random() < 0.5:
        coords[:, 1] *= -1
    for _ in range(random.randrange(4)):
        coords = np.stack((-coords[:, 1], coords[:, 0]), axis=1)
    return _normalize_shape(map(tuple, coords))


def _frame_centers(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
    random.shuffle(candidates)
    chosen = []
    for cand in candidates:
        if all(abs(cand[0] - rr) >= 6 or abs(cand[1] - cc) >= 6 for rr, cc in chosen):
            chosen.append(cand)
            if len(chosen) == count:
                return chosen
    return None


def _paint_frame(grid: np.ndarray, center: tuple[int, int], center_color: int) -> None:
    r, c = center
    for rr, cc in _frame_ring(r, c):
        grid[rr, cc] = GREY
    grid[r, c] = center_color


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in coords}


def _objects_touch(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    set_b = _coords_set(coords_b)
    for r, c in _coords_set(coords_a):
        if (r, c) in set_b:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in set_b:
                return True
    return False


def _candidate_placements(
    h: int,
    w: int,
    shape: np.ndarray,
    frames: list[dict],
    frame_idx: int,
    reserved: np.ndarray,
    halo: np.ndarray,
) -> list[dict]:
    frame = frames[frame_idx]
    direction = frame["direction"]
    shape_h = int(shape[:, 0].max()) + 1
    shape_w = int(shape[:, 1].max()) + 1
    candidates = []

    for top in range(h - shape_h + 1):
        for left in range(w - shape_w + 1):
            coords = shape + np.array([[top, left]], dtype=int)
            rows = coords[:, 0]
            cols = coords[:, 1]
            if np.any(reserved[rows, cols]) or np.any(halo[rows, cols]):
                continue
            assigned = _assign_frame_or_none(coords, frames)
            if assigned is None:
                continue
            nearest_idx, _dist_nums = assigned
            if nearest_idx != frame_idx:
                continue
            if _centroid_distance_num(coords, frame["center"]) > int(coords.shape[0]) * GEN_FRAME_LOCAL_DISTANCE:
                continue
            boundary_dist = _boundary_distance(coords, direction, (h, w))
            if boundary_dist <= 0:
                continue
            candidates.append(
                {
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "boundary_distance": boundary_dist,
                    "dominant_side": _dominant_side(coords, frame["center"]),
                    "nearest_walls": _nearest_wall_dirs(coords, (h, w)),
                }
            )
    random.shuffle(candidates)
    return candidates


def _pair_geometry_ok(coords_a: np.ndarray, coords_b: np.ndarray, direction: tuple[int, int]) -> bool:
    rows_a = set(map(int, coords_a[:, 0]))
    rows_b = set(map(int, coords_b[:, 0]))
    cols_a = set(map(int, coords_a[:, 1]))
    cols_b = set(map(int, coords_b[:, 1]))
    box_a = _bbox(coords_a)
    box_b = _bbox(coords_b)
    if direction in {(0, 1), (0, -1)}:
        return bool(rows_a & rows_b) and (box_a[3] < box_b[1] or box_b[3] < box_a[1])
    return bool(cols_a & cols_b) and (box_a[2] < box_b[0] or box_b[2] < box_a[0])


def _mark_object(coords: np.ndarray, reserved: np.ndarray, halo: np.ndarray) -> None:
    h, w = reserved.shape
    for r, c in coords:
        reserved[int(r), int(c)] = True
    for r, c in coords:
        r = int(r)
        c = int(c)
        halo[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                halo[nr, nc] = True


def _place_objects(
    h: int,
    w: int,
    frames: list[dict],
    object_count: int,
) -> list[np.ndarray] | None:
    reserved = np.zeros((h, w), dtype=bool)
    halo = np.zeros((h, w), dtype=bool)
    for frame in frames:
        r, c = frame["center"]
        for rr, cc in _frame_ring(r, c):
            reserved[rr, cc] = True
        reserved[r, c] = True

    blocker_order = list(range(len(frames)))
    random.shuffle(blocker_order)

    pair = None
    blocker_idx = None
    used_boundary_distances: set[int] = set()
    placed: list[np.ndarray] = []

    for candidate_frame_idx in blocker_order:
        direction = frames[candidate_frame_idx]["direction"]
        for _ in range(30):
            shape_a = _random_shape()
            shape_b = _random_shape()
            cand_a = _candidate_placements(h, w, shape_a, frames, candidate_frame_idx, reserved, halo)
            cand_b = _candidate_placements(h, w, shape_b, frames, candidate_frame_idx, reserved, halo)
            if not cand_a or not cand_b:
                continue
            for item_a in cand_a[:80]:
                for item_b in cand_b[:80]:
                    if item_a["boundary_distance"] == item_b["boundary_distance"]:
                        continue
                    if not _pair_geometry_ok(item_a["coords"], item_b["coords"], direction):
                        continue
                    if _objects_touch(item_a["coords"], item_b["coords"]):
                        continue
                    pair = (item_a["coords"], item_b["coords"])
                    blocker_idx = candidate_frame_idx
                    break
                if pair is not None:
                    break
            if pair is not None:
                break
        if pair is not None:
            break

    if pair is None or blocker_idx is None:
        return None

    for coords in pair:
        used_boundary_distances.add(_boundary_distance(coords, frames[blocker_idx]["direction"], (h, w)))
        _mark_object(coords, reserved, halo)
        placed.append(coords)

    remaining = object_count - 2
    other_indices = [i for i in range(len(frames)) if i != blocker_idx]
    assigned_frames = []
    if remaining > 0:
        assigned_frames.append(random.choice(other_indices))
    while len(assigned_frames) < remaining:
        assigned_frames.append(random.randrange(len(frames)))
    random.shuffle(assigned_frames)

    need_counterexample = True
    for frame_idx in assigned_frames:
        direction_name = DIR_NAME[frames[frame_idx]["direction"]]
        chosen = None
        for _ in range(40):
            shape = _random_shape()
            candidates = _candidate_placements(h, w, shape, frames, frame_idx, reserved, halo)
            candidates = [item for item in candidates if item["boundary_distance"] not in used_boundary_distances]
            if not candidates:
                continue
            preferred = [
                item
                for item in candidates
                if item["dominant_side"] != direction_name and direction_name not in item["nearest_walls"]
            ]
            if preferred and need_counterexample:
                chosen = random.choice(preferred)
            else:
                chosen = random.choice(candidates)
            break
        if chosen is None:
            return None
        coords = chosen["coords"]
        used_boundary_distances.add(chosen["boundary_distance"])
        _mark_object(coords, reserved, halo)
        placed.append(coords)
        if chosen["dominant_side"] != direction_name and direction_name not in chosen["nearest_walls"]:
            need_counterexample = False

    random.shuffle(placed)
    return placed


def _safe_distractor_cells(grid: np.ndarray, frame_count: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            trial = grid.copy()
            trial[r, c] = GREY
            if len(_find_frames(trial)) == frame_count:
                cells.append((r, c))
    return cells


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        frames = _find_frames(grid)
        if not (3 <= len(frames) <= 4):
            return False

        objects = _find_orange_objects(grid)
        if not (3 <= len(objects) <= 5):
            return False
        if any(not (3 <= int(obj.shape[0]) <= 5) for obj in objects):
            return False

        orange_count = int(np.count_nonzero(grid == ORANGE))
        if orange_count != sum(int(obj.shape[0]) for obj in objects):
            return False

        frame_cells = set()
        centers = set()
        for frame in frames:
            centers.add(frame["center"])
            frame_cells.add(frame["center"])
            frame_cells.update(_frame_ring(*frame["center"]))
        grey_distractors = [
            (int(r), int(c))
            for r, c in np.argwhere(grid == GREY)
            if (int(r), int(c)) not in frame_cells
        ]
        if not grey_distractors:
            return False

        out, _frames, results = _simulate(grid)
        if not np.array_equal(out, main(grid)):
            return False
        if set(map(int, np.unique(out))) - {BG, ORANGE}:
            return False
        if int(np.count_nonzero(out == ORANGE)) != orange_count:
            return False

        directions = {result["direction_name"] for result in results}
        if len(directions) < 2:
            return False

        boundary_dists = [int(result["boundary_distance"]) for result in results]
        if len(boundary_dists) != len(set(boundary_dists)):
            return False

        if not any(int(result["shift"]) > 0 for result in results):
            return False
        if not any(result["stop_reason"] == "object" for result in results):
            return False
        if not any(result["stop_reason"] == "wall" for result in results):
            return False

        if any(_head_on(results[i], results[j]) for i in range(len(results)) for j in range(i + 1, len(results))):
            return False

        if not any(result["direction_name"] not in result["nearest_walls"] for result in results):
            return False
        if not any(result["direction_name"] != result["dominant_side"] for result in results):
            return False

        return True
    except Exception:
        return False


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    frame_count = random.randint(*GEN_FRAME_COUNT_RANGE)
    object_count = random.randint(*GEN_OBJECT_COUNT_RANGE)

    centers = _frame_centers(h, w, frame_count)
    if centers is None:
        return None

    center_colors = random.sample(list(DIR_BY_CENTER), frame_count)
    frames = [
        {
            "center": center,
            "center_color": color,
            "direction": DIR_BY_CENTER[color],
        }
        for center, color in zip(centers, center_colors)
    ]

    objects = _place_objects(h, w, frames, object_count)
    if objects is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        _paint_frame(grid, frame["center"], frame["center_color"])
    for coords in objects:
        grid[coords[:, 0], coords[:, 1]] = ORANGE

    distractor_target = random.randint(*GEN_DISTRACTOR_COUNT_RANGE)
    for _ in range(distractor_target):
        safe = _safe_distractor_cells(grid, frame_count)
        if not safe:
            break
        r, c = random.choice(safe)
        grid[r, c] = GREY

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-direction sliding puzzle")
