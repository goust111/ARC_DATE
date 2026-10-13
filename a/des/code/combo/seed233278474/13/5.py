from __future__ import annotations

import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_RETRIES = 300
OBJECT_COUNT_RANGE = (4, 8)
QUADRANT_SIZE_RANGE = (7, 14)

DIRECTIONS = ("up", "right", "left", "down")
MISSING_BY_DIRECTION = {
    "up": (0, 1),
    "right": (1, 2),
    "left": (1, 0),
    "down": (2, 1),
}
QUADRANT_BY_DIRECTION = {
    "up": (0, 0),
    "right": (0, 1),
    "left": (1, 0),
    "down": (1, 1),
}
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _shape_mask(direction: str) -> np.ndarray:
    mask = np.ones((3, 3), dtype=bool)
    rr, cc = MISSING_BY_DIRECTION[direction]
    mask[rr, cc] = False
    return mask


SHAPE_MASKS = {direction: _shape_mask(direction) for direction in DIRECTIONS}


def _paint_object(grid: np.ndarray, top: int, left: int, direction: str, color: int) -> None:
    mask = SHAPE_MASKS[direction]
    patch = grid[top : top + 3, left : left + 3]
    patch[mask] = int(color)


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == 0:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == 0:
                        continue
                    assert int(grid[nr, nc]) == color, "each object must be monochrome"
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            pts = np.asarray(coords, dtype=int)
            r0 = int(np.min(pts[:, 0]))
            c0 = int(np.min(pts[:, 1]))
            r1 = int(np.max(pts[:, 0]))
            c1 = int(np.max(pts[:, 1]))
            objects.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return objects


def _classify_direction(obj: dict) -> str:
    coords = np.asarray(obj["coords"], dtype=int)
    r0, c0, r1, c1 = obj["bbox"]
    assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "each object must have a 3x3 bbox"

    local = np.zeros((3, 3), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    assert int(np.count_nonzero(local)) == 8, "each object must contain 8 pixels"

    missing = np.argwhere(~local)
    assert missing.shape == (1, 2), "each object must miss exactly one cell"
    key = tuple(int(v) for v in missing[0])
    for direction, cell in MISSING_BY_DIRECTION.items():
        if key == cell:
            return direction
    raise AssertionError("missing cell must be the center pixel of one side")


def _pack_count_fits(count: int, qh: int, qw: int) -> bool:
    if count == 0:
        return True
    if qh < 3 or qw < 3:
        return False
    cols = (qw + 1) // 4
    if cols < 1:
        return False
    rows = (count + cols - 1) // cols
    return 4 * rows - 1 <= qh


def _render_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    h, w = shape
    assert h % 2 == 1 and w % 2 == 1, "grid must have odd height and width"

    qh = h // 2
    qw = w // 2
    out = np.zeros((h, w), dtype=int)

    buckets = {direction: [] for direction in DIRECTIONS}
    for spec in specs:
        direction = str(spec["direction"])
        assert direction in buckets
        buckets[direction].append(int(spec["color"]))

    origins = {
        (0, 0): (0, 0),
        (0, 1): (0, qw + 1),
        (1, 0): (qh + 1, 0),
        (1, 1): (qh + 1, qw + 1),
    }

    for direction in DIRECTIONS:
        colors = buckets[direction]
        if not colors:
            continue

        assert _pack_count_fits(len(colors), qh, qw), "quadrant is too small for packed objects"
        cols = (qw + 1) // 4
        base_r, base_c = origins[QUADRANT_BY_DIRECTION[direction]]
        for idx, color in enumerate(colors):
            rr = idx // cols
            cc = idx % cols
            _paint_object(out, base_r + 4 * rr, base_c + 4 * cc, direction, color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Group 3x3 C-shaped objects by opening direction and repack them into fixed quadrants."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _find_objects(g)
    ordered = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    specs = []
    for obj in ordered:
        specs.append(
            {
                "direction": _classify_direction(obj),
                "color": int(obj["color"]),
            }
        )
    return _render_output(g.shape, specs)


def _sample_directions(count: int) -> list[str]:
    assert count >= 4
    if random.random() < 0.5:
        directions = list(DIRECTIONS)
        while len(directions) < count:
            directions.append(random.choice(DIRECTIONS))
        random.shuffle(directions)
        return directions

    while True:
        directions = [random.choice(DIRECTIONS) for _ in range(count)]
        if len(set(directions)) >= 2:
            return directions


def _choose_shape(counts: dict[str, int]) -> tuple[int, int] | None:
    qmin, qmax = QUADRANT_SIZE_RANGE
    candidates = []
    for qh in range(qmin, qmax + 1):
        for qw in range(qmin, qmax + 1):
            if all(_pack_count_fits(counts[direction], qh, qw) for direction in DIRECTIONS):
                candidates.append((qh, qw))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0] * item[1])
    roomy = candidates[len(candidates) // 3 :]
    return random.choice(roomy or candidates)


def _place_random_objects(shape: tuple[int, int], specs: list[dict]) -> np.ndarray | None:
    h, w = shape
    if h < 3 or w < 3:
        return None

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    order = specs[:]
    random.shuffle(order)

    for spec in order:
        direction = str(spec["direction"])
        color = int(spec["color"])
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, h - 3)
            left = random.randint(0, w - 3)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(h, top + 4)
            c1 = min(w, left + 4)
            if np.any(blocked[r0:r1, c0:c1]):
                continue

            _paint_object(grid, top, left, direction, color)
            blocked[r0:r1, c0:c1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _expected_specs(directions: list[str], color: int) -> list[dict]:
    return [{"direction": direction, "color": int(color)} for direction in directions]


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    directions = _sample_directions(n_objects)
    counts = {direction: directions.count(direction) for direction in DIRECTIONS}

    quadrant_shape = _choose_shape(counts)
    if quadrant_shape is None:
        return None
    qh, qw = quadrant_shape
    shape = (2 * qh + 1, 2 * qw + 1)

    color = random.randint(1, 9)
    specs = _expected_specs(directions, color)
    grid = _place_random_objects(shape, specs)
    if grid is None:
        return None

    expected = _render_output(shape, specs)
    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
