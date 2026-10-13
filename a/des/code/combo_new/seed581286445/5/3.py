import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MIN_OBJECTS = 3
MAX_OBJECTS = 6
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_points(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _canonical_shape_key(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rel = _normalize_points(points)
    max_r = max(r for r, _ in rel)
    max_c = max(c for _, c in rel)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in rel:
        mask[r, c] = True

    keys = []
    rotated = mask
    for _ in range(4):
        coords = np.argwhere(rotated)
        keys.append(tuple(sorted((int(r), int(c)) for r, c in coords)))
        rotated = np.rot90(rotated)
    return min(keys)


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            points: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                points.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            r0, c0, r1, c1 = _bbox(points)
            objects.append(
                {
                    "color": color,
                    "points": points,
                    "bbox": (r0, c0, r1, c1),
                    "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
                    "shape_key": _canonical_shape_key(points),
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor equal-shape objects by cycling colors through their bounding-box order.

    Input structure:
    - The background is black.
    - Several 4-connected monochrome foreground objects are scattered in the grid.
    - Every foreground object has the same shape up to rotation, but each object uses a different color.

    Rule:
    - Find every foreground object and compute the area of its axis-aligned bounding box.
    - Sort the objects by bounding-box area from small to large; when areas tie, sort by the top row of the box.
    - Cyclically shift the sorted color list one step to the right, then recolor each object in place with its assigned new color.

    Key insight:
    - The objects must be grouped by shared shape under rotation, while the transfer order comes from the enclosing rectangle rather than from pixel count or from the colors themselves.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    assert len(objects) >= 2, "expected at least two foreground objects"

    shape_keys = {obj["shape_key"] for obj in objects}
    assert len(shape_keys) == 1, "all objects should share one shape up to rotation"

    ranked = sorted(objects, key=lambda obj: (int(obj["bbox_area"]), int(obj["bbox"][0]), int(obj["bbox"][1])))
    shifted_colors = [int(ranked[-1]["color"])] + [int(obj["color"]) for obj in ranked[:-1]]

    out = g.copy()
    for obj, new_color in zip(ranked, shifted_colors):
        for r, c in obj["points"]:
            out[r, c] = new_color
    return out


def _random_shape_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h, w = rng.choice(((2, 3), (2, 4), (3, 4), (4, 3)))
        if rng.random() < 0.5:
            h, w = w, h

        min_area = max(4, h + w - 2)
        max_area = min(h * w - 1, h + w + 1)
        if min_area > max_area:
            continue
        target_area = rng.randint(min_area, max_area)

        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        if len(cells) != target_area:
            continue

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if not ({0, h - 1} <= rows and {0, w - 1} <= cols):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if np.all(mask):
            continue
        return mask

    raise RuntimeError("failed to build a non-degenerate shape")


def _rotated_points(mask: np.ndarray, turns: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    rotated = np.rot90(mask, turns)
    points = [(int(r), int(c)) for r, c in np.argwhere(rotated)]
    return points, tuple(int(v) for v in rotated.shape)


def _sample_orientations(rng: random.Random, count: int) -> list[int]:
    orientations = [rng.choice((0, 2)), rng.choice((1, 3))]
    while len(orientations) < count:
        orientations.append(rng.randrange(4))
    rng.shuffle(orientations)
    return orientations


def _can_place(
    blocked: np.ndarray,
    points: list[tuple[int, int]],
    origin: tuple[int, int],
) -> bool:
    h, w = blocked.shape
    r0, c0 = origin
    for dr, dc in points:
        rr, cc = r0 + dr, c0 + dc
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if blocked[rr, cc]:
            return False
    return True


def _paint_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    points: list[tuple[int, int]],
    origin: tuple[int, int],
    color: int,
) -> None:
    h, w = grid.shape
    r0, c0 = origin
    for dr, dc in points:
        rr, cc = r0 + dr, c0 + dc
        grid[rr, cc] = color
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
    shape_mask = _random_shape_mask(rng)
    colors = rng.sample(PALETTE, count)
    orientations = _sample_orientations(rng, count)
    variants = []
    for turns in orientations:
        points, (vh, vw) = _rotated_points(shape_mask, turns)
        variants.append({"points": points, "height": vh, "width": vw})

    max_h = max(int(v["height"]) for v in variants)
    max_w = max(int(v["width"]) for v in variants)
    min_h = max(10, max_h + 4, count * 3 + 2)
    min_w = max(10, max_w + 4, count * 2 + 3)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)
    used_top_rows: set[int] = set()

    order = list(range(count))
    rng.shuffle(order)
    for idx in order:
        variant = variants[idx]
        points = variant["points"]
        vh = int(variant["height"])
        vw = int(variant["width"])
        placed = False
        for _ in range(250):
            r0 = rng.randint(0, grid_h - vh)
            c0 = rng.randint(0, grid_w - vw)
            if r0 in used_top_rows:
                continue
            if not _can_place(blocked, points, (r0, c0)):
                continue
            _paint_object(grid, blocked, points, (r0, c0), colors[idx])
            used_top_rows.add(r0)
            placed = True
            break
        if not placed:
            return None

    objects = _extract_objects(grid)
    if len(objects) != count:
        return None
    if len({obj["color"] for obj in objects}) != count:
        return None
    if len({obj["shape_key"] for obj in objects}) != 1:
        return None
    if len({obj["bbox"][0] for obj in objects}) != count:
        return None
    if len({(obj["bbox"][2] - obj["bbox"][0] + 1, obj["bbox"][3] - obj["bbox"][1] + 1) for obj in objects}) < 2:
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
