from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
TRAIL = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIRECTION_ORDER = ("top", "left", "bottom", "right")

GRID_SIZE_RANGE = (12, 19)  # high exclusive
RECT_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_RANGE = (1, 4)  # inclusive
MIN_DISTANCE = 2
MAX_DISTANCE = 6
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _randint_inclusive(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _foreground_rectangles(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
    rectangles: list[tuple[tuple[int, int, int, int], int]] = []
    for coords in _components4(grid != BG):
        color = int(grid[coords[0]])
        assert all(int(grid[r, c]) == color for r, c in coords), "components must be monochrome"
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "foreground objects must be solid rectangles"
        rectangles.append((bbox, color))
    rectangles.sort(key=lambda item: item[0])
    return rectangles


def _edge_distances(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> dict[str, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return {
        "top": int(r0),
        "left": int(c0),
        "bottom": int(h - 1 - r1),
        "right": int(w - 1 - c1),
    }


def _nearest_edge(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    distances = _edge_distances(bbox, shape)
    return min(DIRECTION_ORDER, key=lambda direction: distances[direction])


def _target_bbox(
    bbox: tuple[int, int, int, int], direction: str, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    obj_h = r1 - r0 + 1
    obj_w = c1 - c0 + 1
    if direction == "top":
        return 0, c0, obj_h - 1, c1
    if direction == "bottom":
        return h - obj_h, c0, h - 1, c1
    if direction == "left":
        return r0, 0, r1, obj_w - 1
    if direction == "right":
        return r0, w - obj_w, r1, w - 1
    raise ValueError(direction)


def _trail_coords(
    start_bbox: tuple[int, int, int, int], end_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = start_bbox
    nr0, nc0, nr1, nc1 = end_bbox
    coords: list[tuple[int, int]] = []
    if nr0 < r0:
        for r in range(nr1 + 1, r0):
            for c in range(c0, c1 + 1):
                coords.append((r, c))
    elif nr0 > r0:
        for r in range(r1 + 1, nr0):
            for c in range(c0, c1 + 1):
                coords.append((r, c))
    elif nc0 < c0:
        for c in range(nc1 + 1, c0):
            for r in range(r0, r1 + 1):
                coords.append((r, c))
    elif nc0 > c0:
        for c in range(c1 + 1, nc0):
            for r in range(r0, r1 + 1):
                coords.append((r, c))
    return coords


def _motion_mask(
    shape: tuple[int, int],
    start_bbox: tuple[int, int, int, int],
    end_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = start_bbox
    nr0, nc0, nr1, nc1 = end_bbox
    if c0 == nc0:
        mask[min(r0, nr0) : max(r1, nr1) + 1, c0 : c1 + 1] = True
    else:
        mask[r0 : r1 + 1, min(c0, nc0) : max(c1, nc1) + 1] = True
    return mask


def _boxes_too_close(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every solid foreground rectangle to its nearest edge and leave a grey trail behind it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _foreground_rectangles(g)
    if not rectangles:
        return g.copy()

    out = np.full_like(g, BG)
    shape = g.shape
    placements: list[tuple[tuple[int, int, int, int], int]] = []

    for bbox, color in rectangles:
        direction = _nearest_edge(bbox, shape)
        target_bbox = _target_bbox(bbox, direction, shape)
        for r, c in _trail_coords(bbox, target_bbox):
            if int(g[r, c]) == BG:
                out[r, c] = TRAIL
        placements.append((target_bbox, color))

    for bbox, color in placements:
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    return out


def _choose_with_preference(
    rng: np.random.Generator,
    preferred: tuple[int, int],
    fallback: tuple[int, int],
) -> int | None:
    plow, phigh = preferred
    if plow <= phigh:
        return _randint_inclusive(rng, plow, phigh)
    flow, fhigh = fallback
    if flow <= fhigh:
        return _randint_inclusive(rng, flow, fhigh)
    return None


def _sample_candidate(
    rng: np.random.Generator,
    shape: tuple[int, int],
    direction: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = shape
    max_obj_h = min(RECT_SIDE_RANGE[1], h - 2 * MIN_DISTANCE - 1)
    max_obj_w = min(RECT_SIDE_RANGE[1], w - 2 * MIN_DISTANCE - 1)
    if max_obj_h < RECT_SIDE_RANGE[0] or max_obj_w < RECT_SIDE_RANGE[0]:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        obj_h = _randint_inclusive(rng, RECT_SIDE_RANGE[0], max_obj_h)
        obj_w = _randint_inclusive(rng, RECT_SIDE_RANGE[0], max_obj_w)

        if direction in ("top", "bottom"):
            max_distance = min(
                MAX_DISTANCE,
                (h - obj_h - 1) // 2,
                (w - obj_w - 2) // 2,
            )
        else:
            max_distance = min(
                MAX_DISTANCE,
                (w - obj_w - 1) // 2,
                (h - obj_h - 2) // 2,
            )
        if max_distance < MIN_DISTANCE:
            continue
        distance = _randint_inclusive(rng, MIN_DISTANCE, max_distance)

        if direction == "top":
            r0 = distance
            c_low = distance + 1
            c_high = w - obj_w - distance - 1
            c0 = _choose_with_preference(
                rng,
                (c_low, min(c_high, w // 2 - obj_w)),
                (c_low, c_high),
            )
        elif direction == "bottom":
            r0 = h - obj_h - distance
            c_low = distance + 1
            c_high = w - obj_w - distance - 1
            c0 = _choose_with_preference(
                rng,
                (max(c_low, w // 2), c_high),
                (c_low, c_high),
            )
        elif direction == "left":
            c0 = distance
            r_low = distance + 1
            r_high = h - obj_h - distance - 1
            r0 = _choose_with_preference(
                rng,
                (max(r_low, h // 2), r_high),
                (r_low, r_high),
            )
        elif direction == "right":
            c0 = w - obj_w - distance
            r_low = distance + 1
            r_high = h - obj_h - distance - 1
            r0 = _choose_with_preference(
                rng,
                (r_low, min(r_high, h // 2 - obj_h)),
                (r_low, r_high),
            )
        else:
            raise ValueError(direction)

        if r0 is None or c0 is None:
            continue

        bbox = (int(r0), int(c0), int(r0 + obj_h - 1), int(c0 + obj_w - 1))
        distances = _edge_distances(bbox, shape)
        min_distance = min(distances.values())
        if sum(value == min_distance for value in distances.values()) != 1:
            continue
        if _nearest_edge(bbox, shape) != direction:
            continue

        return bbox, _target_bbox(bbox, direction, shape)

    return None


def _scene_is_valid(grid: np.ndarray) -> bool:
    rectangles = _foreground_rectangles(grid)
    if len(rectangles) < 2:
        return False
    colors = [color for _, color in rectangles]
    if len(set(colors)) != len(colors):
        return False
    directions = []
    for bbox, _ in rectangles:
        distances = _edge_distances(bbox, grid.shape)
        min_distance = min(distances.values())
        if sum(value == min_distance for value in distances.values()) != 1:
            return False
        directions.append(_nearest_edge(bbox, grid.shape))
    return len(set(directions)) == len(directions)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        h = int(rng.integers(*GRID_SIZE_RANGE))
        w = int(rng.integers(*GRID_SIZE_RANGE))
        grid = np.full((h, w), BG, dtype=int)

        count = int(rng.choice(np.array(RECT_COUNT_CHOICES)))
        directions = list(rng.choice(np.array(DIRECTION_ORDER), size=count, replace=False))
        rng.shuffle(directions)
        colors = list(map(int, rng.choice(PALETTE, size=count, replace=False)))

        objects: list[dict[str, object]] = []
        occupied_motion = np.zeros((h, w), dtype=bool)
        ok = True

        for direction, color in zip(directions, colors):
            placed = False
            for _ in range(PLACEMENT_ATTEMPTS):
                candidate = _sample_candidate(rng, (h, w), str(direction))
                if candidate is None:
                    continue
                bbox, target_bbox = candidate
                if any(_boxes_too_close(bbox, obj["bbox"]) for obj in objects):
                    continue
                motion = _motion_mask((h, w), bbox, target_bbox)
                if np.any(occupied_motion & motion):
                    continue

                objects.append(
                    {
                        "bbox": bbox,
                        "color": int(color),
                        "motion": motion,
                    }
                )
                occupied_motion |= motion
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(obj["color"])

        if _scene_is_valid(grid):
            return grid

    raise RuntimeError("generate_input failed to build a valid sliding-rectangles scene")
