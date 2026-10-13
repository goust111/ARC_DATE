from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_objects(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    objs: list[dict] = []
    for coords in _components_mask(grid != bg):
        r0, c0, r1, c1 = _bbox_coords(coords)
        sprite = _crop_bbox(grid, (r0, c0, r1, c1))
        colors = {int(v) for v in sprite[sprite != bg]}
        assert len(colors) == 1, "each sprite must be monochrome"
        color = next(iter(colors))
        area = int(coords.shape[0])
        digit = area % 10
        assert 1 <= digit <= 9, "sprite area must have a non-zero units digit"
        objs.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "color": int(color),
                "area": area,
                "digit": digit,
            }
        )

    assert 3 <= len(objs) <= 5, "expected 3-5 sprites"
    areas = [obj["area"] for obj in objs]
    assert len(set(areas)) == len(areas), "sprite areas must be distinct"
    colors = [obj["color"] for obj in objs]
    assert len(set(colors)) == len(colors), "sprite colors must be distinct"
    return objs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild scattered sprites as concentric square layers ordered by sprite area.

    Input structure:
    - A black background contains 3-5 scattered monochrome 4-connected sprites.
    - Each sprite uses a distinct non-black color and has a distinct pixel area.
    - Sprites are separated from one another and only their color and area matter.

    Rule:
    - Extract each sprite, count its pixel area, and sort the sprites by area from small to large.
    - The smallest sprite becomes the innermost solid square.
    - Wrap outward in sorted order with concentric square frames whose thickness is the corresponding sprite area units digit; for valid task instances this is the intended layer size signal.
    - Color each layer with the color of the sprite that supplied it and return the tight square output with no black padding.

    Key insight:
    - Each sprite's area carries two roles at once: its total area determines layer order, while its units digit determines the size of the layer that color contributes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = sorted(_extract_objects(g), key=lambda obj: obj["area"])

    inner = int(objects[0]["digit"])
    current = np.full((inner, inner), int(objects[0]["color"]), dtype=int)
    for obj in objects[1:]:
        thick = int(obj["digit"])
        next_side = int(current.shape[0] + 2 * thick)
        wrapped = np.full((next_side, next_side), int(obj["color"]), dtype=int)
        wrapped[thick : thick + current.shape[0], thick : thick + current.shape[1]] = current
        current = wrapped
    return current


_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_AREA_POOL = [3, 4, 11, 12, 13, 14]
_COUNT_WEIGHTS = [(3, 3), (4, 4), (5, 3)]
_GRID_RANGE = (12, 16)
_MAX_ATTEMPTS = 400

_SMALL_SHAPES = {
    3: [
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
    ],
    4: [
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    ],
}


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=int)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.rot90(out, k=random.randrange(4))


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_blob(area: int) -> np.ndarray | None:
    if area in _SMALL_SHAPES:
        return _random_transform(random.choice(_SMALL_SHAPES[area]))

    for _ in range(160):
        h = random.randint(3, min(6, area - 1))
        w = random.randint(3, min(6, area - 1))
        if h * w <= area:
            continue

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {
            nbr for nbr in _neighbors4(*start) if 0 <= nbr[0] < h and 0 <= nbr[1] < w
        }

        while len(cells) < area and frontier:
            options = list(frontier)
            weights = []
            for r, c in options:
                neighbors_in = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
                weights.append(3 if neighbors_in >= 2 else 1)
            r, c = random.choices(options, weights=weights, k=1)[0]
            frontier.discard((r, c))
            if (r, c) in cells:
                continue
            cells.add((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=int)
        for r, c in cells:
            mask[r, c] = 1
        mask = _tight_crop_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        return _random_transform(mask)
    return None


def _sample_areas() -> list[int]:
    counts, weights = zip(*_COUNT_WEIGHTS)
    count = random.choices(counts, weights=weights, k=1)[0]
    areas = random.sample(_AREA_POOL, k=count)
    areas.sort()
    return areas


def _shape_for_area(area: int) -> np.ndarray | None:
    return _random_blob(area)


def _reserve_clear(reserved: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(reserved.shape[0], r + h + 1)
    c1 = min(reserved.shape[1], c + w + 1)
    return not bool(np.any(reserved[r0:r1, c0:c1]))


def _place_shapes(shapes: list[np.ndarray], h: int, w: int) -> list[tuple[int, int]] | None:
    reserved = np.zeros((h, w), dtype=bool)
    indexed = sorted(enumerate(shapes), key=lambda item: item[1].size, reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)

    for idx, shape in indexed:
        sh, sw = shape.shape
        options = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        random.shuffle(options)
        placed = False
        for r, c in options:
            if not _reserve_clear(reserved, r, c, sh, sw):
                continue
            reserved[r : r + sh, c : c + sw] = True
            placements[idx] = (r, c)
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in placements if pos is not None]


def _render_input(shapes: list[np.ndarray], colors: list[int], positions: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for shape, color, (r0, c0) in zip(shapes, colors, positions):
        rr, cc = np.nonzero(shape)
        grid[r0 + rr, c0 + cc] = int(color)
    return grid


def _output_side_from_areas(areas: list[int]) -> int:
    digits = [area % 10 for area in sorted(areas)]
    return int(digits[0] + 2 * sum(digits[1:]))


def _gen_attempt() -> np.ndarray | None:
    areas = _sample_areas()
    out_side = _output_side_from_areas(areas)
    if out_side > 30:
        return None

    shapes: list[np.ndarray] = []
    for area in areas:
        shape = _shape_for_area(area)
        if shape is None:
            return None
        shapes.append(shape)

    grid_h = random.randint(*_GRID_RANGE)
    grid_w = random.randint(*_GRID_RANGE)
    if max(shape.shape[0] for shape in shapes) > grid_h or max(shape.shape[1] for shape in shapes) > grid_w:
        return None

    positions = _place_shapes(shapes, grid_h, grid_w)
    if positions is None:
        return None

    colors = random.sample(_PALETTE, k=len(shapes))
    grid = _render_input(shapes, colors, positions, grid_h, grid_w)

    try:
        objs = _extract_objects(grid)
        if sorted(obj["area"] for obj in objs) != areas:
            return None
        if not np.array_equal(main(grid), main(grid)):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
