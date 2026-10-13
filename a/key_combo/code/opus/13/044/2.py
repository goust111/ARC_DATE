from __future__ import annotations

from collections import deque
import random

import numpy as np


PALETTE = (1, 2, 4, 6, 7, 8)
AREA_VALUES = tuple(range(3, 9))
GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (12, 20)
OBJECT_COUNT_RANGE = (4, 6)
GEN_RETRY_LIMIT = 500
PLACE_RETRY_LIMIT = 250
INCLUDE_HOLE_PROB = 0.68
SOLID_EIGHT_PROB = 0.85


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _transform(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set(cells)
    for _ in range(random.randint(0, 3)):
        pts = {(c, -r) for r, c in pts}
    if random.random() < 0.5:
        pts = {(r, -c) for r, c in pts}
    if random.random() < 0.5:
        pts = {(-r, c) for r, c in pts}
    return _normalize(pts)


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _patch_from_cells(cells: set[tuple[int, int]], color: int = 1) -> np.ndarray:
    h, w = _shape_dims(cells)
    patch = np.zeros((h, w), dtype=int)
    for r, c in cells:
        patch[r, c] = color
    return patch


def _bbox_has_hole(patch: np.ndarray) -> bool:
    h, w = patch.shape
    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if patch[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if patch[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and patch[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any((patch == 0) & (~visited)))


def _component_bfs(grid: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> np.ndarray:
    color = int(grid[start])
    q: deque[tuple[int, int]] = deque([start])
    seen[start] = True
    coords: list[tuple[int, int]] = []
    h, w = grid.shape

    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))

    return np.array(coords, dtype=int)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == 0:
                continue
            coords = _component_bfs(g, (r, c), seen)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": int(g[r, c]),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    comps = _find_components(grid)
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        comp["hollow"] = _bbox_has_hole((patch != 0).astype(int))
    return comps


def _build_output(items: list[dict]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: (item["area"], item["color"], item["bbox"][0], item["bbox"][1]))
    max_area = max(item["area"] for item in ordered)
    out = np.zeros((max_area * 2 + 1, len(ordered)), dtype=int)
    divider = max_area

    for col, item in enumerate(ordered):
        area = int(item["area"])
        color = int(item["color"])
        if item["hollow"]:
            out[divider + 1 : divider + 1 + area, col] = color
        else:
            out[divider - area : divider, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Classify scattered sprites by enclosed holes, then stack area bars upward or downward by class.

    Input structure:
    - The grid has black background and 4-6 separated single-color sprites.
    - Each sprite is a 4-connected component with a unique visible color and area 3-8.
    - Some sprites are solid, while others contain a black region enclosed inside their own bounding box.

    Rule:
    - For each sprite, inspect its bounding box and flood from every black border cell within that box.
    - If any black cell inside the box remains unreachable, classify the sprite as hollow; otherwise classify it as solid.
    - Compute each sprite's pixel area, sort sprites by area from small to large, and assign one output column per sprite.
    - Draw a vertical divider row of black at the middle of the output. Solid sprites become upward bars of their own color with height equal to area; hollow sprites become downward bars of their own color with height equal to area.

    Key insight:
    - The same measured quantity, area, determines bar height and left-to-right order, but structural hole detection determines the growth direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    items = _analyze_objects(g)
    assert items
    return _build_output(items)


_SOLID_LIBRARY: dict[int, list[set[tuple[int, int]]]] = {
    3: [
        {(0, 0), (0, 1), (1, 0)},
        {(0, 0), (1, 0), (2, 0)},
    ],
    4: [
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 0), (0, 1), (0, 2), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (3, 0)},
    ],
    5: [
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
        {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)},
    ],
    6: [
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)},
        {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)},
        {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)},
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)},
    ],
    7: [
        {(r, c) for r in range(3) for c in range(3)} - {(0, 2), (2, 0)},
        {(r, c) for r in range(3) for c in range(3)} - {(0, 0), (0, 2)},
        {(r, c) for r in range(3) for c in range(3)} - {(0, 0), (2, 2)},
    ],
    8: [
        {(r, c) for r in range(3) for c in range(3)} - {(0, 1)},
        {(r, c) for r in range(3) for c in range(3)} - {(0, 0)},
        {(r, c) for r in range(3) for c in range(3)} - {(2, 1)},
        {(r, c) for r in range(2) for c in range(4)},
    ],
}

_HOLLOW_RING = {(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)}


def _sample_solid_shape(area: int) -> set[tuple[int, int]]:
    for _ in range(40):
        if area == 8 and random.random() < 0.75:
            choices = [
                {(r, c) for r in range(3) for c in range(3)} - {(0, 1)},
                {(r, c) for r in range(3) for c in range(3)} - {(1, 0)},
                {(r, c) for r in range(3) for c in range(3)} - {(1, 2)},
                {(r, c) for r in range(3) for c in range(3)} - {(2, 1)},
            ]
            cells = _transform(random.choice(choices))
        else:
            cells = _transform(set(random.choice(_SOLID_LIBRARY[area])))
        if not _bbox_has_hole((_patch_from_cells(cells, 1) != 0).astype(int)):
            return cells
    raise RuntimeError(f"failed to sample solid shape for area {area}")


def _sample_shape(area: int, hollow: bool) -> set[tuple[int, int]]:
    if hollow:
        return _transform(set(_HOLLOW_RING))
    return _sample_solid_shape(area)


def _sample_area_plan() -> tuple[list[tuple[int, bool]], bool]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    include_hole = random.random() < INCLUDE_HOLE_PROB

    if include_hole:
        other_areas = random.sample([3, 4, 5, 6, 7], count - 1)
        plan = [(8, True)] + [(area, False) for area in other_areas]
    else:
        if random.random() < SOLID_EIGHT_PROB:
            other_areas = random.sample([3, 4, 5, 6, 7], count - 1)
            areas = [8] + other_areas
        else:
            areas = random.sample(list(AREA_VALUES), count)
        plan = [(area, False) for area in areas]

    random.shuffle(plan)
    return plan, include_hole


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    return not bool(np.any(grid[rr0:rr1, cc0:cc1] != 0))


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != 0
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _sample_layout() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = np.zeros((H, W), dtype=int)

    plan, include_hole = _sample_area_plan()
    colors = random.sample(list(PALETTE), len(plan))

    objects: list[dict] = []
    for (area, hollow), color in zip(plan, colors):
        cells = _sample_shape(area, hollow)
        patch = _patch_from_cells(cells, color=color)
        objects.append(
            {
                "area": area,
                "hollow": hollow,
                "color": color,
                "patch": patch,
                "size": int((patch != 0).sum()),
            }
        )

    objects.sort(key=lambda item: (item["patch"].shape[0] * item["patch"].shape[1], item["area"]), reverse=True)

    for obj in objects:
        patch = obj["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(PLACE_RETRY_LIMIT):
            r0 = random.randint(0, H - h)
            c0 = random.randint(0, W - w)
            if _can_place(grid, patch, r0, c0):
                _place_patch(grid, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None

    if not _semantic_check(grid, expect_hole=include_hole):
        return None
    return grid


def _semantic_check(grid: np.ndarray, expect_hole: bool | None = None) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    items = _analyze_objects(g)
    if not (4 <= len(items) <= 6):
        return False

    colors = [item["color"] for item in items]
    areas = [item["area"] for item in items]
    hollows = [bool(item["hollow"]) for item in items]

    if len(set(colors)) != len(colors):
        return False
    if len(set(areas)) != len(areas):
        return False
    if any(area < 3 or area > 8 for area in areas):
        return False
    if expect_hole is True and not any(hollows):
        return False
    if expect_hole is False and any(hollows):
        return False

    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if g[r, c] == 0:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] != 0 and g[nr, nc] != g[r, c]:
                    return False

    out = main(g)
    expected = _build_output(items)
    if not np.array_equal(out, expected):
        return False

    max_area = max(areas)
    divider = max_area
    if out.shape != (max_area * 2 + 1, len(items)):
        return False
    if not np.all(out[divider, :] == 0):
        return False

    ordered = sorted(items, key=lambda item: (item["area"], item["color"], item["bbox"][0], item["bbox"][1]))
    for col, item in enumerate(ordered):
        area = item["area"]
        color = item["color"]
        if item["hollow"]:
            if not np.all(out[: divider + 1, col] == 0):
                return False
            if not np.all(out[divider + 1 : divider + 1 + area, col] == color):
                return False
            if not np.all(out[divider + 1 + area :, col] == 0):
                return False
        else:
            if not np.all(out[: divider - area, col] == 0):
                return False
            if not np.all(out[divider - area : divider, col] == color):
                return False
            if not np.all(out[divider:, col] == 0):
                return False

    if not any(hollows) and 8 in areas:
        area8 = next(item for item in items if item["area"] == 8)
        if area8["hollow"]:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _sample_layout()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
