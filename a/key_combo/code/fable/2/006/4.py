from __future__ import annotations

from collections import Counter
import random

import numpy as np


_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_GEN_BRANCHES = ("smallest", "largest", "middle")
_GEN_CALL_COUNT = 0
_GEN_MAX_ATTEMPTS = 400
_GEN_MIN_OBJECTS = 4
_GEN_MAX_OBJECTS = 8
_GEN_MAX_SIDE = 30
_GEN_MIN_SIDE = 16
_GEN_GAP = 1
_GEN_SIZE_SLACK = 10
_GEN_SHAPE_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, *, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return objects


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    objects = _find_components(g, bg=bg)
    assert objects, "expected at least one foreground object"

    count = len(objects)
    areas = [int(obj["area"]) for obj in objects]
    assert len(areas) == len(set(areas)), "all object areas must be distinct"

    pointer_candidates = [obj for obj in objects if int(obj["area"]) == count]
    assert len(pointer_candidates) == 1, "expected a unique area-equals-count pointer object"
    pointer = pointer_candidates[0]

    color_counts = Counter(int(obj["color"]) for obj in objects)
    pointer_color = int(pointer["color"])
    assert color_counts[pointer_color] == 2, "pointer color must appear on exactly two objects"
    assert sum(cnt == 2 for cnt in color_counts.values()) == 1, "only the pointer color may repeat"
    assert all(cnt == 1 for color, cnt in color_counts.items() if color != pointer_color)

    same_color = [obj for obj in objects if int(obj["color"]) == pointer_color]
    assert len(same_color) == 2
    target = same_color[1] if same_color[0] is pointer else same_color[0]
    return {
        "bg": int(bg),
        "count": int(count),
        "objects": objects,
        "pointer": pointer,
        "target": target,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique area-equals-count object as a color pointer, then crop the other object of that color.

    Input structure:
    - A black background contains several monochrome 4-connected objects.
    - All foreground object areas are distinct.
    - Exactly one object has area equal to the total number of foreground objects.
    - Exactly one other object shares that pointer object's color; every remaining object has its own color.

    Rule:
    - Count the foreground objects to get N.
    - Find the unique object whose area is N; that object is only the pointer.
    - Find the other object with the same color as the pointer object.
    - Return that second object's tight bounding-box crop, preserving its shape and color.

    Key insight:
    - The self-referential area clue identifies a color, not the final answer; one more same-color indirection step is required.
    """
    parsed = _parse_scene(np.asarray(grid, dtype=int))
    return _crop_bbox(grid, parsed["target"]["bbox"])


def _patch_dim_choices(area: int) -> list[tuple[int, int]]:
    limit = min(max(2, area), 7)
    choices = []
    for h in range(1, limit + 1):
        for w in range(1, limit + 1):
            box_area = h * w
            if area <= box_area <= area + 5:
                choices.append((h, w))
    return choices


def _random_connected_patch(area: int, color: int) -> np.ndarray:
    for _ in range(_GEN_SHAPE_ATTEMPTS):
        h, w = random.choice(_patch_dim_choices(area))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        coords = np.asarray(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), _BG, dtype=int)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)

        k = random.randrange(4)
        if k:
            patch = np.rot90(patch, k)
        if random.random() < 0.5:
            patch = patch[::-1, :]
        if random.random() < 0.5:
            patch = patch[:, ::-1]
        return patch
    raise RuntimeError(f"failed to build a connected patch with area {area}")


def _choose_target_area(other_areas: list[int], relation: str) -> int:
    sorted_areas = sorted(int(a) for a in other_areas)
    if relation in {"smallest", "largest"} and len(sorted_areas) >= 3:
        return int(random.choice(sorted_areas[1:-1]))
    return int(random.choice(sorted_areas))


def _sample_areas(count: int, relation: str) -> tuple[int, int, list[int]]:
    pointer_area = int(count)
    if relation == "largest":
        other_areas = list(range(1, count))
    elif relation == "smallest":
        pool = list(range(count + 1, count + 12))
        other_areas = random.sample(pool, count - 1)
    else:
        n_smaller = random.randint(1, count - 2)
        n_larger = count - 1 - n_smaller
        smaller = random.sample(list(range(1, count)), n_smaller)
        larger = random.sample(list(range(count + 1, count + 12)), n_larger)
        other_areas = smaller + larger

    target_area = _choose_target_area(other_areas, relation)
    distractor_areas = [int(area) for area in other_areas if int(area) != target_area]
    random.shuffle(distractor_areas)
    return pointer_area, target_area, distractor_areas


def _sample_objects(relation: str) -> list[dict]:
    count = random.randint(_GEN_MIN_OBJECTS, _GEN_MAX_OBJECTS)
    pointer_area, target_area, distractor_areas = _sample_areas(count, relation)
    pointer_color = int(random.choice(_COLOR_POOL))
    other_colors = random.sample([c for c in _COLOR_POOL if c != pointer_color], count - 2)

    objects = [
        {
            "role": "pointer",
            "color": pointer_color,
            "area": pointer_area,
            "patch": _random_connected_patch(pointer_area, pointer_color),
        },
        {
            "role": "target",
            "color": pointer_color,
            "area": target_area,
            "patch": _random_connected_patch(target_area, pointer_color),
        },
    ]

    for color, area in zip(other_colors, distractor_areas):
        objects.append(
            {
                "role": "distractor",
                "color": int(color),
                "area": int(area),
                "patch": _random_connected_patch(int(area), int(color)),
            }
        )

    return objects


def _candidate_locations(canvas: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = canvas.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return []

    filled = np.argwhere(patch != _BG)
    locs = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            ok = True
            for pr, pc in filled:
                rr = r0 + int(pr)
                cc = c0 + int(pc)
                r1 = max(0, rr - _GEN_GAP)
                r2 = min(h, rr + _GEN_GAP + 1)
                c1 = max(0, cc - _GEN_GAP)
                c2 = min(w, cc + _GEN_GAP + 1)
                if np.any(canvas[r1:r2, c1:c2] != _BG):
                    ok = False
                    break
            if ok:
                locs.append((r0, c0))
    return locs


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != _BG
    canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _grid_size(objects: list[dict]) -> tuple[int, int]:
    total_area = sum(int(obj["area"]) for obj in objects)
    tallest = max(int(obj["patch"].shape[0]) for obj in objects)
    widest = max(int(obj["patch"].shape[1]) for obj in objects)
    side = max(_GEN_MIN_SIDE, int(np.ceil(np.sqrt(total_area * 6.0))), tallest + 4, widest + 4)
    upper = min(_GEN_MAX_SIDE, side + _GEN_SIZE_SLACK)
    return random.randint(side, upper), random.randint(side, upper)


def _place_objects(objects: list[dict]) -> np.ndarray | None:
    h, w = _grid_size(objects)
    canvas = np.full((h, w), _BG, dtype=int)
    order = list(range(len(objects)))
    random.shuffle(order)
    order.sort(key=lambda idx: (-int(objects[idx]["area"]), idx))

    for idx in order:
        locs = _candidate_locations(canvas, np.asarray(objects[idx]["patch"], dtype=int))
        if not locs:
            return None
        r0, c0 = random.choice(locs)
        objects[idx]["loc"] = (int(r0), int(c0))
        _paste(canvas, np.asarray(objects[idx]["patch"], dtype=int), int(r0), int(c0))
    return canvas


def _pointer_relation(parsed: dict) -> str:
    areas = sorted(int(obj["area"]) for obj in parsed["objects"])
    pointer_area = int(parsed["pointer"]["area"])
    if pointer_area == areas[0]:
        return "smallest"
    if pointer_area == areas[-1]:
        return "largest"
    return "middle"


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_scene(grid)
    return _crop_bbox(grid, parsed["target"]["bbox"])


def _validate_sample(grid: np.ndarray, relation: str) -> bool:
    try:
        parsed = _parse_scene(grid)
        assert int(parsed["bg"]) == _BG
        assert _pointer_relation(parsed) == relation
        expected = _independent_expected(grid)
        actual = main(grid)
        assert np.array_equal(actual, expected)

        pointer_crop = _crop_bbox(grid, parsed["pointer"]["bbox"])
        target_crop = _crop_bbox(grid, parsed["target"]["bbox"])
        assert not np.array_equal(pointer_crop, target_crop)
        assert expected.shape == target_crop.shape
        return True
    except Exception:
        return False


def _gen_attempt(relation: str) -> np.ndarray | None:
    objects = _sample_objects(relation)
    grid = _place_objects(objects)
    if grid is None:
        return None
    if not _validate_sample(grid, relation):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    primary_relation = _GEN_BRANCHES[_GEN_CALL_COUNT % len(_GEN_BRANCHES)]
    _GEN_CALL_COUNT += 1
    relation_order = [primary_relation] + [rel for rel in _GEN_BRANCHES if rel != primary_relation]

    for relation in relation_order:
        for _ in range(_GEN_MAX_ATTEMPTS // len(relation_order)):
            grid = _gen_attempt(relation)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid self-referential pointer scene")
