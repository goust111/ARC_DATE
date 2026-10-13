from __future__ import annotations

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))
RNG = np.random.default_rng()

# Generator configuration.
GRID_SIDE_MIN = 16
GRID_SIDE_MAX_EXCLUSIVE = 27
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX_EXCLUSIVE = 6
AREA_MIN = 1
AREA_MAX_EXCLUSIVE = 13
MAX_SHAPE_SIDE = 7
PLACEMENT_PADDING = 1
MAX_SHAPE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _mask_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize(coords)
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _coords_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _component_perimeter(coords: set[tuple[int, int]]) -> int:
    perimeter = 0
    for r, c in coords:
        for dr, dc in DIR4:
            if (r + dr, c + dc) not in coords:
                perimeter += 1
    return perimeter


def _has_hole(coords: set[tuple[int, int]]) -> bool:
    mask = _mask_from_coords(coords)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(height):
        for c in (0, width - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(width):
        for r in (0, height - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any(~mask & ~seen))


def _is_rectangle(coords: set[tuple[int, int]]) -> bool:
    mask = _mask_from_coords(coords)
    return int(mask.sum()) == int(mask.shape[0] * mask.shape[1])


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, int(RNG.integers(0, 4)))
    if bool(RNG.integers(0, 2)):
        out = np.fliplr(out)
    return out


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for r0 in range(height):
        for c0 in range(width):
            color = int(g[r0, c0])
            if color == bg or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coord_set = set(coords)
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            components.append(
                {
                    "color": color,
                    "coords": coord_set,
                    "area": len(coord_set),
                    "perimeter": _component_perimeter(coord_set),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold each object's clockwise outer contour into a horizontal segment.

    Because every foreground object is monochrome, preserving the contour-pixel
    colors reduces to keeping the object color and using the contour length as the
    segment length. For 4-connected hole-free polyominoes, that contour length is
    the usual 4-neighbor perimeter: one unit per exposed side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    objects = _find_components(g, bg)
    assert objects, "expected at least one foreground object"

    objects.sort(key=lambda obj: (int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1]), int(obj["color"])))
    max_len = max(int(obj["perimeter"]) for obj in objects)

    out = np.full((len(objects), max_len), int(bg), dtype=int)
    for row, obj in enumerate(objects):
        out[row, : int(obj["perimeter"])] = int(obj["color"])
    return out


def _random_polyomino(area: int) -> set[tuple[int, int]] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        coords = {(0, 0)}
        while len(coords) < area:
            frontier = set()
            center_r = sum(r for r, _ in coords) / len(coords)
            center_c = sum(c for _, c in coords) / len(coords)
            choices: list[tuple[int, int]] = []
            weights: list[float] = []

            for r, c in coords:
                for dr, dc in DIR4:
                    cell = (r + dr, c + dc)
                    if cell in coords or cell in frontier:
                        continue
                    trial = _normalize(coords | {cell})
                    trial_mask = _mask_from_coords(trial)
                    if max(trial_mask.shape) > MAX_SHAPE_SIDE:
                        continue
                    frontier.add(cell)
                    neighbors = sum((cell[0] + rr, cell[1] + cc) in coords for rr, cc in DIR4)
                    distance = abs(cell[0] - center_r) + abs(cell[1] - center_c)
                    choices.append(cell)
                    weights.append(((neighbors + 1) ** 2) / (distance + 1.0))

            if not choices:
                break
            pick = int(RNG.choice(len(choices), p=np.asarray(weights) / np.sum(weights)))
            coords.add(choices[pick])

        if len(coords) != area:
            continue
        mask = _transform_mask(_mask_from_coords(coords))
        shape = _coords_from_mask(mask)
        if _has_hole(shape):
            continue
        return _normalize(shape)
    return None


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    height, width = occupied.shape
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > height or c0 + w > width:
        return False

    for dr, dc in np.argwhere(mask):
        r = int(r0 + dr)
        c = int(c0 + dc)
        r1 = max(0, r - padding)
        r2 = min(height, r + padding + 1)
        c1 = max(0, c - padding)
        c2 = min(width, c + padding + 1)
        if bool(np.any(occupied[r1:r2, c1:c2])):
            return False
    return True


def _place_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        grid[rr, cc] = int(color)
        occupied[rr, cc] = True


def _expected_output(bg: int, specs: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(specs, key=lambda spec: (int(spec["area"]), int(spec["order"])))
    width = max(int(spec["perimeter"]) for spec in ordered)
    out = np.full((len(ordered), width), int(bg), dtype=int)
    for row, spec in enumerate(ordered):
        out[row, : int(spec["perimeter"])] = int(spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(GRID_SIDE_MIN, GRID_SIDE_MAX_EXCLUSIVE))
    width = int(RNG.integers(GRID_SIDE_MIN, GRID_SIDE_MAX_EXCLUSIVE))
    n_obj = int(RNG.integers(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX_EXCLUSIVE))

    bg = int(RNG.choice(PALETTE))
    colors = [int(v) for v in RNG.choice([c for c in PALETTE if c != bg], size=n_obj, replace=False)]
    areas = sorted(int(v) for v in RNG.choice(np.arange(AREA_MIN, AREA_MAX_EXCLUSIVE), size=n_obj, replace=False))

    masks: list[np.ndarray] = []
    specs: list[dict[str, object]] = []
    irregular_found = False
    for order, (area, color) in enumerate(zip(areas, colors)):
        shape = _random_polyomino(area)
        if shape is None:
            return None
        mask = _mask_from_coords(shape)
        masks.append(mask)
        perimeter = _component_perimeter(shape)
        irregular_found = irregular_found or not _is_rectangle(shape)
        specs.append(
            {
                "order": order,
                "area": area,
                "color": color,
                "perimeter": perimeter,
                "mask": mask,
            }
        )

    if not irregular_found and n_obj >= 3:
        return None

    if sum(areas) * 2 >= height * width:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(n_obj), key=lambda i: int(masks[i].shape[0] * masks[i].shape[1]), reverse=True)

    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        positions = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        if not positions:
            return None
        for pos_idx in RNG.permutation(len(positions)):
            r0, c0 = positions[int(pos_idx)]
            if _can_place(occupied, mask, int(r0), int(c0), PLACEMENT_PADDING):
                _place_mask(grid, occupied, mask, int(specs[idx]["color"]), int(r0), int(c0))
                break
        else:
            return None

    if int(_get_bg(grid)) != bg:
        return None

    found = _find_components(grid, bg)
    if len(found) != n_obj:
        return None
    if len({int(obj["area"]) for obj in found}) != n_obj:
        return None
    if any(_has_hole(set(obj["coords"])) for obj in found):
        return None

    expected = _expected_output(bg, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
