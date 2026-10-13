from __future__ import annotations

import math
import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_SIDE = 30
GEN_RETRIES = 400
TEMP_CANVAS = 25
TEMP_CENTER = TEMP_CANVAS // 2
EDGE_MARGIN = 1

_SQUARE_AREAS = (4, 9, 16)
_COMPOSITE_AREAS = (6, 8, 10, 12, 14, 15, 18, 20)
_STRIP_AREAS = (5, 7, 11, 13)


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _component_points(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []

    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                pt = (int(rr), int(cc))
                if pt not in seen:
                    seen.add(pt)
                    stack.append(pt)
    return list(seen)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_component_points(mask)) == int(coords.shape[0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_points(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _near_square_dims(area: int) -> tuple[int, int]:
    """Return the factor pair closest to a square, with width >= height."""
    for h in range(int(math.isqrt(area)), 0, -1):
        if area % h == 0:
            return int(h), int(area // h)
    raise AssertionError("positive area must have a factorization")


def _centered_top_left(center: tuple[int, int], height: int, width: int) -> tuple[int, int]:
    cr, cc = center
    return int(cr - (height - 1) // 2), int(cc - (width - 1) // 2)


def _ring_around(mask: np.ndarray) -> np.ndarray:
    nbr = np.zeros_like(mask, dtype=bool)
    nbr[1:, :] |= mask[:-1, :]
    nbr[:-1, :] |= mask[1:, :]
    nbr[:, 1:] |= mask[:, :-1]
    nbr[:, :-1] |= mask[:, 1:]
    return nbr & (~mask)


def _rebuild_layers(
    shape: tuple[int, int],
    center: tuple[int, int],
    layers: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    occupied = np.zeros(shape, dtype=bool)

    for index, (color, area) in enumerate(layers):
        color = int(color)
        area = int(area)
        if index == 0:
            height, width = _near_square_dims(area)
            top, left = _centered_top_left(center, height, width)
            assert top >= 0 and left >= 0
            assert top + height <= shape[0] and left + width <= shape[1]
            out[top : top + height, left : left + width] = color
            occupied[top : top + height, left : left + width] = True
            continue

        ring = _ring_around(occupied)
        assert ring.any(), "each outer color must have a background ring to occupy"
        out[ring] = color
        occupied |= ring

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder a compact multi-color object into concentric layers sorted by color area.

    Input structure:
    - The background is black.
    - A single compact foreground object is formed by 2 to 4 colors and is 4-connected overall.
    - Each foreground color occupies its own 4-connected region, and the color areas are all distinct.

    Rule:
    - Find the center pixel of the foreground object's bounding box.
    - Sort the foreground colors by their pixel counts from largest to smallest.
    - Clear the original object and rebuild it on the same center: the largest color becomes the innermost solid rectangle whose area matches that color count and whose dimensions are the closest possible to a square, with width at least height.
    - Every remaining color becomes a one-pixel-thick outer contour ring by filling all background cells that are orthogonally adjacent to the already rebuilt inner structure.

    Key insight:
    - The original arrangement is discarded completely; only each color's area rank matters, which determines the order of the new nested center-to-outside structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg = g != BG
    assert fg.any(), "expected a foreground object"
    assert _is_connected(fg), "foreground must form one 4-connected object"

    coords = np.argwhere(fg)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    center = ((r0 + r1) // 2, (c0 + c1) // 2)

    layers: list[tuple[int, int]] = []
    for color in sorted(int(v) for v in np.unique(g[fg])):
        mask = g == color
        assert _is_connected(mask), "each color must form one 4-connected layer"
        layers.append((int(color), int(mask.sum())))

    areas = [area for _color, area in layers]
    assert len(set(areas)) == len(areas), "color areas must be distinct"
    layers.sort(key=lambda item: int(item[1]), reverse=True)

    return _rebuild_layers(g.shape, center, layers)


def _choose_largest_area(color_count: int) -> int:
    branch = random.choice(("square", "composite", "strip"))
    if branch == "square":
        options = [area for area in _SQUARE_AREAS if area >= color_count]
    elif branch == "composite":
        options = [area for area in _COMPOSITE_AREAS if area >= color_count]
    else:
        options = [area for area in _STRIP_AREAS if area >= color_count]
    assert options, "area branch must have at least one legal option"
    return int(random.choice(options))


def _choose_areas(color_count: int) -> list[int]:
    for _ in range(200):
        largest = _choose_largest_area(color_count)
        smaller = random.sample(list(range(1, largest)), color_count - 1)
        areas = [largest] + sorted((int(v) for v in smaller), reverse=True)
        total = sum(areas)
        height, width = _near_square_dims(total)
        if height >= 2 and width >= 2 and max(height, width) <= 10:
            return areas
    raise RuntimeError("failed to choose compact distinct areas")


def _frontier_from_points(points: set[tuple[int, int]], occupied: np.ndarray) -> set[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in points:
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < occupied.shape[0] and 0 <= cc < occupied.shape[1] and not occupied[rr, cc]:
                frontier.add((int(rr), int(cc)))
    return frontier


def _pick_seed(occupied: np.ndarray, occupied_points: set[tuple[int, int]]) -> tuple[int, int] | None:
    candidates = _frontier_from_points(occupied_points, occupied)
    if not candidates:
        return None

    bbox = _bbox_from_points(occupied_points)
    scored = []
    for cell in candidates:
        rr, cc = cell
        expand = _bbox_area(_bbox_from_points(occupied_points | {cell})) - _bbox_area(bbox)
        dist = abs(rr - TEMP_CENTER) + abs(cc - TEMP_CENTER)
        touch = sum(
            1
            for nr, nc in _neighbors4(rr, cc)
            if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]
        )
        scored.append((expand, dist, -touch, cell))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    pool = [cell for *_rest, cell in scored[: min(5, len(scored))]]
    return random.choice(pool)


def _pick_growth_cell(
    region: set[tuple[int, int]],
    occupied: np.ndarray,
    occupied_points: set[tuple[int, int]],
) -> tuple[int, int] | None:
    frontier = _frontier_from_points(region, occupied)
    if not frontier:
        return None

    base_points = occupied_points | region
    base_bbox = _bbox_from_points(base_points)
    scored = []
    for cell in frontier:
        rr, cc = cell
        expand = _bbox_area(_bbox_from_points(base_points | {cell})) - _bbox_area(base_bbox)
        adj_region = sum((nr, nc) in region for nr, nc in _neighbors4(rr, cc))
        adj_union = sum(
            1
            for nr, nc in _neighbors4(rr, cc)
            if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]
        )
        dist = abs(rr - TEMP_CENTER) + abs(cc - TEMP_CENTER)
        scored.append((expand, -adj_region, -adj_union, dist, cell))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    pool = [cell for *_rest, cell in scored[: min(6, len(scored))]]
    return random.choice(pool)


def _grow_region(
    size: int,
    occupied: np.ndarray,
    occupied_points: set[tuple[int, int]],
    first: bool,
) -> set[tuple[int, int]] | None:
    if first:
        if occupied[TEMP_CENTER, TEMP_CENTER]:
            return None
        seed = (TEMP_CENTER, TEMP_CENTER)
    else:
        seed = _pick_seed(occupied, occupied_points)
        if seed is None:
            return None

    region = {seed}
    while len(region) < size:
        cell = _pick_growth_cell(region, occupied, occupied_points)
        if cell is None:
            return None
        region.add(cell)
    return region


def _render_object(regions: list[tuple[int, set[tuple[int, int]]]]) -> np.ndarray:
    canvas = np.full((TEMP_CANVAS, TEMP_CANVAS), BG, dtype=int)
    for color, region in regions:
        for r, c in region:
            canvas[r, c] = int(color)
    return canvas


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, dtype=int, copy=True)


def _crop_foreground(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)


def _output_extents(areas: list[int]) -> tuple[int, int, int, int]:
    layers = [(idx + 1, area) for idx, area in enumerate(sorted(areas, reverse=True))]
    demo = _rebuild_layers((21, 21), (10, 10), layers)
    coords = np.argwhere(demo != BG)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return 10 - r0, r1 - 10, 10 - c0, c1 - 10


def _near_center_extra(limit: int) -> tuple[int, int]:
    total = random.randint(0, max(0, limit))
    low = max(0, total // 2 - 1)
    high = min(total, total // 2 + 1)
    lead = random.randint(low, high)
    return int(lead), int(total - lead)


def _place_object(grid: np.ndarray, obj: np.ndarray, center: tuple[int, int]) -> None:
    h, w = obj.shape
    top, left = _centered_top_left(center, h, w)
    grid[top : top + h, left : left + w] = obj


def _build_candidate_input() -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    color_count = random.randint(2, 4)
    areas = _choose_areas(color_count)
    colors = random.sample(list(range(1, 10)), color_count)
    color_to_area = list(zip(colors, areas, strict=False))
    total = sum(areas)
    height, width = _near_square_dims(total)
    obj = np.full((height, width), BG, dtype=int)

    cells: list[tuple[int, int]] = []
    for r in range(height):
        cols = range(width) if r % 2 == 0 else range(width - 1, -1, -1)
        for c in cols:
            cells.append((int(r), int(c)))

    order = list(range(color_count))
    random.shuffle(order)
    pos = 0
    for idx in order:
        color, area = color_to_area[idx]
        for r, c in cells[pos : pos + int(area)]:
            obj[r, c] = int(color)
        pos += int(area)

    obj = _random_transform(obj)
    fg = obj != BG
    assert fg.all(), "the generated object should fill its rectangle"
    if not _is_connected(fg):
        return None
    for color, area in color_to_area:
        mask = obj == int(color)
        if int(mask.sum()) != int(area) or not _is_connected(mask):
            return None

    return obj, [(int(color), int(area)) for color, area in color_to_area]


def _assemble_grid(obj: np.ndarray, color_areas: list[tuple[int, int]]) -> np.ndarray | None:
    areas = [area for _color, area in color_areas]
    in_up = (obj.shape[0] - 1) // 2
    in_down = obj.shape[0] - 1 - in_up
    in_left = (obj.shape[1] - 1) // 2
    in_right = obj.shape[1] - 1 - in_left
    out_up, out_down, out_left, out_right = _output_extents(areas)

    need_up = max(in_up, out_up) + EDGE_MARGIN
    need_down = max(in_down, out_down) + EDGE_MARGIN
    need_left = max(in_left, out_left) + EDGE_MARGIN
    need_right = max(in_right, out_right) + EDGE_MARGIN

    min_h = need_up + need_down + 1
    min_w = need_left + need_right + 1
    if min_h > MAX_SIDE or min_w > MAX_SIDE:
        return None

    top_extra, bottom_extra = _near_center_extra(MAX_SIDE - min_h)
    left_extra, right_extra = _near_center_extra(MAX_SIDE - min_w)
    height = min_h + top_extra + bottom_extra
    width = min_w + left_extra + right_extra

    center = (need_up + top_extra, need_left + left_extra)
    grid = np.full((height, width), BG, dtype=int)
    _place_object(grid, obj, center)

    layers = sorted(color_areas, key=lambda item: int(item[1]), reverse=True)
    expected = _rebuild_layers(grid.shape, center, layers)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_candidate_input()
        if built is None:
            continue
        obj, color_areas = built
        grid = _assemble_grid(obj, color_areas)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-layer ARC input")
