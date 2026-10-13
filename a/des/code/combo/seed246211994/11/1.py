from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator settings.
GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (12, 22)
OBJECT_COUNT_RANGE = (2, 5)
AREA_VALUES = tuple(range(1, 10))
INPUT_CLEARANCE = 1
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0, keepdims=True)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _find_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == int(bg):
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "anchor": (
                        int(coord_arr[:, 0].min()),
                        int(coord_arr[:, 1].min()),
                    ),
                }
            )

    objects.sort(key=lambda obj: (obj["anchor"][0], obj["anchor"][1], int(obj["color"])))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every 4-connected monochrome foreground object downward by its own area.

    Pixels translated below the bottom edge are discarded.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = _background_color(g)
    out = np.full_like(g, bg)

    for obj in _find_objects(g, bg):
        color = int(obj["color"])
        coords = np.asarray(obj["coords"], dtype=int)
        shift = int(coords.shape[0])

        moved = coords.copy()
        moved[:, 0] += shift
        moved = moved[moved[:, 0] < h]

        for rr, cc in moved:
            if int(out[rr, cc]) not in (int(bg), color):
                raise AssertionError("shifted objects must not overlap")
            out[rr, cc] = color

    return out


def _transform_shape(coords: np.ndarray, rotations: int, reflect: bool) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    if reflect:
        arr = np.column_stack((arr[:, 0], -arr[:, 1]))
    for _ in range(rotations % 4):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    return _normalize(arr)


def _shape_variants(coords: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    for rotations in range(4):
        for reflect in (False, True):
            variant = _transform_shape(coords, rotations, reflect)
            key = tuple((int(r), int(c)) for r, c in variant)
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant)

    random.shuffle(variants)
    return variants


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier: list[tuple[int, int]] = []
        weights: list[int] = []
        seen: set[tuple[int, int]] = set()

        for r, c in cells:
            for dr, dc in DIRS4:
                cand = (r + dr, c + dc)
                if cand in cells or cand in seen:
                    continue
                seen.add(cand)
                frontier.append(cand)
                adjacency = sum((cand[0] + x, cand[1] + y) in cells for x, y in DIRS4)
                weights.append(max(1, adjacency * adjacency))

        pick = random.choices(frontier, weights=weights, k=1)[0]
        cells.add(pick)

    return _normalize(np.array(sorted(cells), dtype=int))


def _row_choices(grid_h: int, area: int, shape_h: int, mode: str) -> list[int]:
    if mode == "visible":
        hi = int(grid_h) - int(area) - int(shape_h)
        return list(range(0, hi + 1)) if hi >= 0 else []

    if mode != "partial":
        raise ValueError(f"unknown row mode: {mode}")

    lo = max(0, int(grid_h) - int(area) - int(shape_h) + 1)
    hi = int(grid_h) - int(area) - 1
    return list(range(lo, hi + 1)) if hi >= lo else []


def _reserve_cells(mask: np.ndarray, coords: np.ndarray, clearance: int) -> None:
    h, w = mask.shape
    for rr, cc in coords:
        r0 = max(0, int(rr) - int(clearance))
        r1 = min(h - 1, int(rr) + int(clearance))
        c0 = max(0, int(cc) - int(clearance))
        c1 = min(w - 1, int(cc) + int(clearance))
        mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _can_place(mask: np.ndarray, coords: np.ndarray) -> bool:
    return not bool(np.any(mask[coords[:, 0], coords[:, 1]]))


def _choose_modes(specs: list[dict[str, object]], grid_h: int) -> bool:
    visible_capable: list[int] = []
    partial_capable: list[int] = []

    for i, spec in enumerate(specs):
        area = int(spec["area"])
        variants = spec["variants"]
        if any(_row_choices(grid_h, area, int(v[:, 0].max()) + 1, "visible") for v in variants):
            visible_capable.append(i)
        if any(_row_choices(grid_h, area, int(v[:, 0].max()) + 1, "partial") for v in variants):
            partial_capable.append(i)

    mode = random.choices(["all_visible", "mixed"], weights=[2, 3], k=1)[0]
    if mode == "all_visible":
        if len(visible_capable) != len(specs):
            return False
        for spec in specs:
            spec["mode"] = "visible"
        return True

    max_partial = min(len(partial_capable), len(specs) - 1)
    if max_partial == 0:
        return False

    partial_count = random.randint(1, max_partial)
    partial_indices = set(random.sample(partial_capable, partial_count))
    for i, spec in enumerate(specs):
        spec["mode"] = "partial" if i in partial_indices else "visible"
        if spec["mode"] == "visible" and i not in visible_capable:
            return False
    return True


def _place_objects(
    specs: list[dict[str, object]],
    grid_h: int,
    grid_w: int,
) -> list[dict[str, object]] | None:
    input_reserved = np.zeros((grid_h, grid_w), dtype=bool)
    output_occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict[str, object]] = []

    ordered_specs = sorted(
        specs,
        key=lambda spec: (
            0 if spec["mode"] == "partial" else 1,
            -int(spec["area"]),
            -max(int(v[:, 1].max()) + 1 for v in spec["variants"]),
        ),
    )

    for spec in ordered_specs:
        area = int(spec["area"])
        color = int(spec["color"])
        mode = str(spec["mode"])
        found = None

        for shape in spec["variants"]:
            shape_h = int(shape[:, 0].max()) + 1
            shape_w = int(shape[:, 1].max()) + 1
            if shape_w > grid_w:
                continue

            rows = _row_choices(grid_h, area, shape_h, mode)
            if not rows:
                continue

            positions = [(top, left) for top in rows for left in range(0, grid_w - shape_w + 1)]
            random.shuffle(positions)

            for top, left in positions:
                coords = shape + np.array([top, left], dtype=int)
                if not _can_place(input_reserved, coords):
                    continue

                moved = coords.copy()
                moved[:, 0] += area
                visible = moved[moved[:, 0] < grid_h]
                if visible.size == 0:
                    continue
                if bool(np.any(output_occupied[visible[:, 0], visible[:, 1]])):
                    continue

                found = {
                    "color": color,
                    "area": area,
                    "mode": mode,
                    "coords": coords,
                    "moved": visible,
                }
                break

            if found is not None:
                break

        if found is None:
            return None

        _reserve_cells(input_reserved, np.asarray(found["coords"], dtype=int), INPUT_CLEARANCE)
        moved = np.asarray(found["moved"], dtype=int)
        output_occupied[moved[:, 0], moved[:, 1]] = True
        placed.append(found)

    return placed


def _build_specs(grid_h: int, grid_w: int) -> list[dict[str, object]] | None:
    max_objects = min(int(OBJECT_COUNT_RANGE[1]), max(int(OBJECT_COUNT_RANGE[0]), int(grid_w) // 5 + 1))
    if max_objects < int(OBJECT_COUNT_RANGE[0]):
        return None

    count = random.randint(int(OBJECT_COUNT_RANGE[0]), max_objects)
    area_pool = [area for area in AREA_VALUES if area < grid_h]
    if len(area_pool) < count:
        return None

    bg = random.randrange(10)
    colors = random.sample([color for color in range(10) if color != bg], count)
    areas = random.sample(area_pool, count)

    specs: list[dict[str, object]] = []
    for color, area in zip(colors, areas):
        base = _random_polyomino(int(area))
        specs.append(
            {
                "background": bg,
                "color": int(color),
                "area": int(area),
                "variants": _shape_variants(base),
            }
        )

    return specs


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)

    specs = _build_specs(grid_h, grid_w)
    if specs is None or not _choose_modes(specs, grid_h):
        return None

    placed = _place_objects(specs, grid_h, grid_w)
    if placed is None:
        return None

    bg = int(specs[0]["background"])
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    expected = np.full_like(grid, bg)
    clipped_count = 0

    for item in placed:
        color = int(item["color"])
        coords = np.asarray(item["coords"], dtype=int)
        moved = np.asarray(item["moved"], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = color
        expected[moved[:, 0], moved[:, 1]] = color
        if moved.shape[0] < coords.shape[0]:
            clipped_count += 1

    planned_count = len(specs)
    actual_objects = _find_objects(grid, bg)
    if len(actual_objects) != planned_count:
        return None

    actual_areas = sorted(int(obj["coords"].shape[0]) for obj in actual_objects)
    planned_areas = sorted(int(spec["area"]) for spec in specs)
    if actual_areas != planned_areas or len(set(planned_areas)) != planned_count:
        return None

    mixed = any(str(spec["mode"]) == "partial" for spec in specs)
    if mixed:
        if not (1 <= clipped_count < planned_count):
            return None
    elif clipped_count != 0:
        return None

    bg_count = int(np.count_nonzero(grid == bg))
    if bg_count <= max(planned_areas):
        return None
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
