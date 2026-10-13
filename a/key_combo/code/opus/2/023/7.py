from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _bbox_area(box: BBox) -> int:
    return _bbox_h(box) * _bbox_w(box)


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            box = _bbox_coords(coords_arr)
            comps.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": box,
                    "area": int(coords_arr.shape[0]),
                }
            )

    return comps


def _sorted_components(components: list[dict]) -> list[dict]:
    return sorted(components, key=lambda comp: (-int(comp["area"]), comp["bbox"][0], comp["bbox"][1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten each foreground region into a centered horizontal segment, then stack the segments from longest to shortest.

    Input structure:
    - The grid is a black background with several separated non-black foreground regions.
    - Each foreground region is a 4-connected monochrome object.
    - The regions can have different colors, positions, and irregular shapes.

    Rule:
    - Find every 4-connected monochrome foreground region and count its pixels.
    - Replace each region by a horizontal segment of the same color whose length equals that pixel count.
    - Sort the segments by length from longest to shortest; if lengths tie, the region whose topmost pixel is higher comes first.
    - Build an output grid whose height is the number of regions and whose width is the longest segment, then place one centered segment per row.

    Key insight:
    - The original geometry of each region is irrelevant after straightening; only the pixel count survives, and that count determines both the segment length and the row order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g, bg=0)
    assert components, "expected at least one foreground component"

    areas = [int(comp["area"]) for comp in components]
    assert len(set(areas)) == len(areas), "component areas must be unique"

    out_h = len(components)
    out_w = max(areas)
    out = np.zeros((out_h, out_w), dtype=int)

    for out_r, comp in enumerate(_sorted_components(components)):
        length = int(comp["area"])
        color = int(comp["color"])
        left = (out_w - length) // 2
        out[out_r, left : left + length] = color

    return out


GRID_H_RANGE = (11, 16)
GRID_W_RANGE = (11, 16)
COMPONENT_COUNT_RANGE = (3, 5)
AREA_RANGE = (3, 10)
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500
MAX_SHAPE_ATTEMPTS = 120


def _normalize_cells(cells: set[Coord]) -> tuple[Coord, ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_cells(cells: set[Coord]) -> tuple[Coord, ...]:
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    )
    transform = random.choice(transforms)
    return _normalize_cells({transform(r, c) for r, c in cells})


def _shape_patch(cells: tuple[Coord, ...], color: int) -> np.ndarray:
    arr = np.array(cells, dtype=int)
    box = _bbox_coords(arr)
    patch = np.zeros((_bbox_h(box), _bbox_w(box)), dtype=int)
    for r, c in cells:
        patch[int(r), int(c)] = int(color)
    return patch


def _random_shape(area: int) -> tuple[Coord, ...] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        cells: set[Coord] = {(0, 0)}
        while len(cells) < area:
            frontier: set[Coord] = set()
            for r, c in cells:
                for nbr in _neighbors4(r, c):
                    if nbr not in cells:
                        frontier.add(nbr)
            cells.add(random.choice(tuple(frontier)))

        normalized = _transform_cells(cells)
        arr = np.array(normalized, dtype=int)
        box = _bbox_coords(arr)
        if _bbox_h(box) == 1 or _bbox_w(box) == 1:
            continue
        if _bbox_area(box) == area:
            continue
        return normalized
    return None


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False

    for pr, pc in np.argwhere(patch != 0):
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        if int(grid[rr, cc]) != 0:
            return False
        rr0 = max(0, rr - 1)
        rr1 = min(gh, rr + 2)
        cc0 = max(0, cc - 1)
        cc1 = min(gw, cc + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != 0):
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != 0
    window[mask] = patch[mask]


def _is_irregular(comp: dict) -> bool:
    box = comp["bbox"]
    return _bbox_h(box) > 1 and _bbox_w(box) > 1 and int(comp["area"]) < _bbox_area(box)


def _breaks_row_shortcut(components: list[dict]) -> bool:
    return any(
        int(a["area"]) > int(b["area"]) and a["bbox"][0] > b["bbox"][0]
        for a in components
        for b in components
    )


def _breaks_width_shortcut(components: list[dict]) -> bool:
    return any(
        int(a["area"]) > int(b["area"]) and _bbox_w(a["bbox"]) <= _bbox_w(b["bbox"])
        for a in components
        for b in components
    )


def _breaks_bbox_shortcut(components: list[dict]) -> bool:
    return any(
        int(a["area"]) > int(b["area"]) and _bbox_area(a["bbox"]) <= _bbox_area(b["bbox"])
        for a in components
        for b in components
    )


def _semantic_checks(grid: np.ndarray) -> bool:
    components = _find_components(grid, bg=0)
    if not COMPONENT_COUNT_RANGE[0] <= len(components) <= COMPONENT_COUNT_RANGE[1]:
        return False

    areas = [int(comp["area"]) for comp in components]
    if len(set(areas)) != len(areas):
        return False
    if not all(AREA_RANGE[0] <= area <= AREA_RANGE[1] for area in areas):
        return False
    if len({int(comp["color"]) for comp in components}) != len(components):
        return False

    irregular_count = sum(_is_irregular(comp) for comp in components)
    if irregular_count < max(2, len(components) - 1):
        return False
    if not _breaks_row_shortcut(components):
        return False
    if not _breaks_width_shortcut(components):
        return False
    if not _breaks_bbox_shortcut(components):
        return False

    out = main(grid)
    expected_h = len(components)
    expected_w = max(areas)
    if out.shape != (expected_h, expected_w):
        return False

    sorted_components = _sorted_components(components)
    for out_r, comp in enumerate(sorted_components):
        color = int(comp["color"])
        length = int(comp["area"])
        left = (expected_w - length) // 2
        row = out[out_r]
        mask = row != 0
        if int(mask.sum()) != length:
            return False
        if not np.all(row[left : left + length] == color):
            return False
        if np.any(row[:left] != 0) or np.any(row[left + length :] != 0):
            return False

    return True


def _attempt_input() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    count = random.randint(*COMPONENT_COUNT_RANGE)
    areas = random.sample(range(AREA_RANGE[0], AREA_RANGE[1] + 1), count)
    colors = random.sample(PALETTE, count)

    specs = []
    for area, color in zip(areas, colors):
        shape = _random_shape(area)
        if shape is None:
            return None
        patch = _shape_patch(shape, color)
        specs.append({"area": area, "color": color, "patch": patch})

    grid = np.zeros((h, w), dtype=int)
    order = sorted(specs, key=lambda spec: spec["patch"].shape[0] * spec["patch"].shape[1], reverse=True)
    for spec in order:
        patch = spec["patch"]
        ph, pw = patch.shape
        positions = [
            (r0, c0)
            for r0 in range(h - ph + 1)
            for c0 in range(w - pw + 1)
            if _can_place_patch(grid, patch, r0, c0)
        ]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _place_patch(grid, patch, r0, c0)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
