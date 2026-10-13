from __future__ import annotations

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            perimeter = 0
            for rr, cc in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width or int(g[nr, nc]) == bg:
                        perimeter += 1

            objects.append(
                {
                    "color": color,
                    "area": int(len(cells)),
                    "perimeter": int(perimeter),
                    "top": int(coords[:, 0].min()),
                    "left": int(coords[:, 1].min()),
                }
            )

    assert objects, "expected at least one foreground object"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten each foreground object into a horizontal line whose length equals its perimeter,
    then stack the lines from top to bottom in ascending area order.

    Input structure
    - The background is a single dominant color.
    - Foreground pixels form separate 4-connected monochrome objects.

    Rule
    - Compute each object's area and 4-neighbor perimeter.
    - Replace each object with a left-aligned horizontal segment of the same color and
      length equal to that perimeter.
    - Sort objects by area from small to large.
    - Place the segments on a fresh canvas with one background row between consecutive lines.
    - Output width is the longest segment length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    objects = _extract_objects(g, bg)
    objects.sort(key=lambda obj: (int(obj["area"]), int(obj["top"]), int(obj["left"])))

    out_h = 2 * len(objects) - 1
    out_w = max(int(obj["perimeter"]) for obj in objects)
    out = np.full((out_h, out_w), bg, dtype=int)
    for i, obj in enumerate(objects):
        out[2 * i, : int(obj["perimeter"])] = int(obj["color"])
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

BACKGROUND_COLOR = 8
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]

GRID_HEIGHT_RANGE = (10, 19)
GRID_WIDTH_RANGE = (10, 22)
OBJECT_COUNT_RANGE = (2, 6)

AREA_RANGE = (3, 13)
PERIMETER_RANGE = (4, 16)

SHAPE_ATTEMPTS = 120
OBJECT_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _mask_perimeter(mask: np.ndarray) -> int:
    height, width = map(int, mask.shape)
    perimeter = 0
    for r, c in np.argwhere(mask):
        for dr, dc in NEIGHBORS4:
            nr, nc = int(r) + dr, int(c) + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width or not bool(mask[nr, nc]):
                perimeter += 1
    return int(perimeter)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords - coords.min(axis=0)
    out = np.zeros((int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1), dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    if not coords:
        return False
    stack = [coords[0]]
    seen = {coords[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(4)))
    if float(rng.random()) < 0.5:
        out = np.fliplr(out)
    return _trim_mask(out)


def _sample_rectangle_mask(rng: np.random.Generator) -> np.ndarray:
    dims = [
        (h, w)
        for h in range(1, 5)
        for w in range(1, 6)
        if AREA_RANGE[0] <= h * w < AREA_RANGE[1] and PERIMETER_RANGE[0] <= 2 * (h + w) < PERIMETER_RANGE[1]
    ]
    h, w = dims[int(rng.integers(len(dims)))]
    return _random_transform(np.ones((h, w), dtype=bool), rng)


def _sample_l_mask(rng: np.random.Generator) -> np.ndarray:
    shapes: list[np.ndarray] = []
    for arm_h in range(2, 6):
        for arm_w in range(2, 6):
            mask = np.zeros((arm_h, arm_w), dtype=bool)
            mask[:, 0] = True
            mask[-1, :] = True
            area = int(mask.sum())
            perimeter = _mask_perimeter(mask)
            if AREA_RANGE[0] <= area < AREA_RANGE[1] and PERIMETER_RANGE[0] <= perimeter < PERIMETER_RANGE[1]:
                shapes.append(mask)
    return _random_transform(shapes[int(rng.integers(len(shapes)))], rng)


def _sample_t_mask(rng: np.random.Generator) -> np.ndarray:
    shapes: list[np.ndarray] = []
    for width in (3, 5):
        for stem_h in range(2, 5):
            mask = np.zeros((stem_h, width), dtype=bool)
            mask[0, :] = True
            mask[:, width // 2] = True
            area = int(mask.sum())
            perimeter = _mask_perimeter(mask)
            if AREA_RANGE[0] <= area < AREA_RANGE[1] and PERIMETER_RANGE[0] <= perimeter < PERIMETER_RANGE[1]:
                shapes.append(mask)
    return _random_transform(shapes[int(rng.integers(len(shapes)))], rng)


def _sample_trimmed_rectangle_mask(rng: np.random.Generator) -> np.ndarray:
    shapes: list[np.ndarray] = []
    for h in range(2, 5):
        for w in range(2, 6):
            base = np.ones((h, w), dtype=bool)
            patterns = [
                [(0, 0)],
                [(0, w - 1)],
                [(h - 1, 0)],
                [(h - 1, w - 1)],
                [(0, 0), (h - 1, w - 1)],
                [(0, w - 1), (h - 1, 0)],
            ]
            for removed in patterns:
                mask = base.copy()
                for r, c in removed:
                    mask[r, c] = False
                if not _is_connected(mask):
                    continue
                area = int(mask.sum())
                perimeter = _mask_perimeter(mask)
                if AREA_RANGE[0] <= area < AREA_RANGE[1] and PERIMETER_RANGE[0] <= perimeter < PERIMETER_RANGE[1]:
                    shapes.append(mask)
    return _random_transform(shapes[int(rng.integers(len(shapes)))], rng)


def _sample_grown_mask(rng: np.random.Generator) -> np.ndarray | None:
    area = int(rng.integers(*AREA_RANGE))
    side_min = max(2, int(np.ceil(np.sqrt(area))) - 1)
    side_max = min(6, area)
    if side_min > side_max:
        return None

    height = int(rng.integers(side_min, side_max + 1))
    width_min = max(2, int(np.ceil(area / height)))
    width_max = min(6, area)
    if width_min > width_max:
        return None
    width = int(rng.integers(width_min, width_max + 1))
    if height * width < area:
        return None

    seed = (int(rng.integers(height)), int(rng.integers(width)))
    cells = {seed}
    frontier: set[tuple[int, int]] = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = seed[0] + dr, seed[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while len(cells) < area and frontier:
        candidates = list(frontier)
        weights = []
        for r, c in candidates:
            adjacency = sum((r + dr, c + dc) in cells for dr, dc in NEIGHBORS4)
            center_penalty = abs(2 * r - (height - 1)) + abs(2 * c - (width - 1))
            weights.append((adjacency + 1) ** 3 / (center_penalty + 1))
        probs = np.asarray(weights, dtype=float)
        probs /= probs.sum()
        pick = candidates[int(rng.choice(len(candidates), p=probs))]
        frontier.remove(pick)
        cells.add(pick)
        for dr, dc in NEIGHBORS4:
            nr, nc = pick[0] + dr, pick[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != area:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _random_transform(_trim_mask(mask), rng)


def _sample_shape(rng: np.random.Generator) -> np.ndarray | None:
    generators = [
        _sample_rectangle_mask,
        _sample_l_mask,
        _sample_t_mask,
        _sample_trimmed_rectangle_mask,
        _sample_grown_mask,
    ]
    for _ in range(SHAPE_ATTEMPTS):
        mask = generators[int(rng.integers(len(generators)))](rng)
        if mask is None:
            continue
        area = int(mask.sum())
        perimeter = _mask_perimeter(mask)
        if AREA_RANGE[0] <= area < AREA_RANGE[1] and PERIMETER_RANGE[0] <= perimeter < PERIMETER_RANGE[1]:
            return mask
    return None


def _render_expected(objects: list[dict[str, object]], bg: int) -> np.ndarray:
    ordered = sorted(objects, key=lambda obj: (int(obj["area"]), int(obj["index"])))
    out = np.full((2 * len(ordered) - 1, max(int(obj["perimeter"]) for obj in ordered)), bg, dtype=int)
    for i, obj in enumerate(ordered):
        out[2 * i, : int(obj["perimeter"])] = int(obj["color"])
    return out


def _place_objects(objects: list[dict[str, object]], height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    occupied: set[tuple[int, int]] = set()
    order = sorted(
        range(len(objects)),
        key=lambda idx: (objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1], int(objects[idx]["area"])),
        reverse=True,
    )

    for idx in order:
        mask = objects[idx]["mask"]
        cell_offsets = np.argwhere(mask)
        mask_h, mask_w = map(int, mask.shape)
        if mask_h > height - 2 or mask_w > width - 2:
            return None

        placements: list[tuple[int, int]] = []
        for r0 in range(1, height - mask_h):
            for c0 in range(1, width - mask_w):
                ok = True
                for rr, cc in cell_offsets:
                    r, c = r0 + int(rr), c0 + int(cc)
                    for dr, dc in NEIGHBORS8 + [(0, 0)]:
                        if (r + dr, c + dc) in occupied:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    placements.append((r0, c0))
        if not placements:
            return None

        r0, c0 = placements[int(rng.integers(len(placements)))]
        rows = cell_offsets[:, 0] + r0
        cols = cell_offsets[:, 1] + c0
        grid[rows, cols] = int(objects[idx]["color"])
        occupied.update((int(r), int(c)) for r, c in zip(rows, cols))

    return grid


def _sample_object_specs(count: int, height: int, width: int, rng: np.random.Generator) -> list[dict[str, object]] | None:
    objects: list[dict[str, object]] = []
    used_areas: set[int] = set()
    total_area = 0

    for index in range(count):
        accepted = False
        for _ in range(OBJECT_ATTEMPTS):
            mask = _sample_shape(rng)
            if mask is None:
                continue
            area = int(mask.sum())
            perimeter = _mask_perimeter(mask)
            if area in used_areas:
                continue
            if total_area + area > (height * width) // 3:
                continue
            objects.append(
                {
                    "mask": mask,
                    "area": area,
                    "perimeter": perimeter,
                    "index": index,
                }
            )
            used_areas.add(area)
            total_area += area
            accepted = True
            break
        if not accepted:
            return None

    colors = rng.choice(COLOR_POOL, size=count, replace=False)
    for obj, color in zip(objects, colors):
        obj["color"] = int(color)
    return objects


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*GRID_HEIGHT_RANGE))
    width = int(rng.integers(*GRID_WIDTH_RANGE))
    count = int(rng.integers(*OBJECT_COUNT_RANGE))

    objects = _sample_object_specs(count, height, width, rng)
    if objects is None:
        return None

    grid = _place_objects(objects, height, width, rng)
    if grid is None:
        return None
    if int(np.count_nonzero(grid == BACKGROUND_COLOR)) <= sum(int(obj["area"]) for obj in objects):
        return None

    expected = _render_expected(objects, BACKGROUND_COLOR)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(_extract_objects(grid, BACKGROUND_COLOR)) != count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
