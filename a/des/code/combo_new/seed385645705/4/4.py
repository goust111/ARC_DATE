from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0 in range(height):
        for c0 in range(width):
            if not bool(mask[r0, c0]) or bool(seen[r0, c0]):
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and bool(mask[nr, nc])
                        and not bool(seen[nr, nc])
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (int(r0), int(c0))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    packed = np.ascontiguousarray(mask.astype(np.uint8))
    return (tuple(map(int, packed.shape)), packed.tobytes())


def _rotation_keys(mask: np.ndarray) -> list[tuple[tuple[int, int], bytes]]:
    keys: list[tuple[tuple[int, int], bytes]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = _mask_key(rotated)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_rotation_keys(mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered colored fragments into matching holes in one grey template rectangle.

    Input structure:
    - The background is black. One large solid grey rectangle contains several black holes.
    - Outside the rectangle are small monochrome colored fragments; each is a non-black, non-grey 4-connected object.
    - Every fragment color is unique, and each fragment shape matches exactly one hole after a 0, 90, 180, or 270 degree rotation.

    Rule:
    - Find the black hole components inside the grey rectangle.
    - For each outside colored fragment, compare all four rotations of its shape with the hole shapes.
    - Fill the uniquely matching hole with that fragment's color.
    - Remove all outside fragments, keep the grey rectangle pixels unchanged, and leave the rest of the grid black.

    Key insight:
    The colored fragments are shape-coded keys for the holes; the missing step is mentally rotating each fragment before matching it to the template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_points = np.argwhere(g == GREY)
    assert grey_points.size > 0, "expected a grey template rectangle"
    rect_r0, rect_c0 = map(int, grey_points.min(axis=0))
    rect_r1, rect_c1 = map(int, grey_points.max(axis=0))

    template = g[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    assert np.all(template[0, :] == GREY), "template top edge must be grey"
    assert np.all(template[-1, :] == GREY), "template bottom edge must be grey"
    assert np.all(template[:, 0] == GREY), "template left edge must be grey"
    assert np.all(template[:, -1] == GREY), "template right edge must be grey"
    assert not np.any((template != BLACK) & (template != GREY)), "template contains only grey and black"

    hole_infos: list[dict] = []
    hole_by_key: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    rect_h, rect_w = template.shape
    for points in _components(template == BLACK):
        if any(r in (0, rect_h - 1) or c in (0, rect_w - 1) for r, c in points):
            continue
        mask, origin = _component_mask(points)
        idx = len(hole_infos)
        hole_infos.append({"mask": mask, "origin": origin})
        hole_by_key.setdefault(_mask_key(mask), []).append(idx)

    assert hole_infos, "expected at least one hole"

    sprite_mask = (g != BLACK) & (g != GREY)
    sprite_mask[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = False
    sprites = _components(sprite_mask)
    assert len(sprites) == len(hole_infos), "fragments and holes must be one-to-one"

    out = np.full_like(g, BLACK)
    out[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = np.where(
        template == GREY, GREY, BLACK
    )

    filled: set[int] = set()
    for points in sprites:
        colors = {int(g[r, c]) for r, c in points}
        assert len(colors) == 1, "each fragment must be monochrome"
        color = next(iter(colors))

        mask, _origin = _component_mask(points)
        candidates: set[int] = set()
        for key in _rotation_keys(mask):
            candidates.update(hole_by_key.get(key, []))
        assert len(candidates) == 1, "each fragment must match exactly one hole"

        hole_idx = next(iter(candidates))
        assert hole_idx not in filled, "two fragments matched the same hole"
        filled.add(hole_idx)

        hole = hole_infos[hole_idx]
        hmask = hole["mask"]
        hr0, hc0 = hole["origin"]
        target = out[
            rect_r0 + hr0 : rect_r0 + hr0 + hmask.shape[0],
            rect_c0 + hc0 : rect_c0 + hc0 + hmask.shape[1],
        ]
        target[hmask] = int(color)

    assert len(filled) == len(hole_infos), "all holes must be filled"
    return out


# ---------------- generator ----------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
RECT_H_RANGE = (8, 16)
RECT_W_RANGE = (8, 18)
HOLE_COUNT_RANGE = (3, 6)
SHAPE_AREA_RANGE = (4, 8)
SHAPE_BOX_MAX = 5
PLACE_TRIES = 250
GENERATION_ATTEMPTS = 500


def _random_connected_shape() -> np.ndarray | None:
    for _ in range(120):
        box_h = random.randint(2, SHAPE_BOX_MAX)
        box_w = random.randint(2, SHAPE_BOX_MAX)
        max_area = min(SHAPE_AREA_RANGE[1], box_h * box_w)
        if max_area < SHAPE_AREA_RANGE[0]:
            continue
        target = random.randint(SHAPE_AREA_RANGE[0], max_area)

        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        frontier = set(_neighbors4(*start))
        frontier = {
            (r, c) for r, c in frontier if 0 <= r < box_h and 0 <= c < box_w
        }
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target:
            continue
        mask, _ = _component_mask(list(cells))
        if min(mask.shape) < 2:
            continue
        if len(_rotation_keys(mask)) < 2:
            continue
        return mask
    return None


def _sample_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(600):
        if len(shapes) == count:
            return shapes
        mask = _random_connected_shape()
        if mask is None:
            continue
        key = _canonical_rotation_key(mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
    return None


def _can_place_with_padding(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    height, width = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > height or c0 + w > width:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(height, r0 + h + 1)
    cc1 = min(width, c0 + w + 1)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_holes(
    rect_h: int, rect_w: int, shapes: list[np.ndarray]
) -> list[tuple[int, int, np.ndarray]] | None:
    occ = np.zeros((rect_h, rect_w), dtype=bool)
    placements: list[tuple[int, int, np.ndarray]] = []
    order = shapes[:]
    random.shuffle(order)

    for mask in order:
        h, w = mask.shape
        candidates = [
            (r, c)
            for r in range(1, rect_h - h)
            for c in range(1, rect_w - w)
            if _can_place_with_padding(occ, r, c, h, w)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements.append((int(r0), int(c0), mask))
        occ[r0 : r0 + h, c0 : c0 + w] = True

    return placements


def _paint_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = int(color)


def _candidate_locations(occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = occ.shape
    h, w = mask.shape
    return [
        (r, c)
        for r in range(height - h + 1)
        for c in range(width - w + 1)
        if _can_place_with_padding(occ, r, c, h, w)
    ]


def _rotation_plan(count: int) -> list[int]:
    rotations = list(range(4))
    random.shuffle(rotations)
    while len(rotations) < count:
        more = list(range(4))
        random.shuffle(more)
        rotations.extend(more)
    return rotations[:count]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    max_rect_h = min(RECT_H_RANGE[1], height - 4)
    max_rect_w = min(RECT_W_RANGE[1], width - 4)
    if max_rect_h < RECT_H_RANGE[0] or max_rect_w < RECT_W_RANGE[0]:
        return None

    rect_h = random.randint(RECT_H_RANGE[0], max_rect_h)
    rect_w = random.randint(RECT_W_RANGE[0], max_rect_w)
    rect_r0 = random.randint(1, height - rect_h - 1)
    rect_c0 = random.randint(1, width - rect_w - 1)

    hole_count = random.randint(*HOLE_COUNT_RANGE)
    shapes = _sample_shapes(hole_count)
    if shapes is None:
        return None
    placements = _place_holes(rect_h, rect_w, shapes)
    if placements is None:
        return None

    colors = random.sample(PALETTE, hole_count)
    hole_infos = [
        (r, c, mask, color)
        for (r, c, mask), color in zip(placements, colors)
    ]

    grid = np.full((height, width), BLACK, dtype=int)
    grid[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = GREY
    expected = grid.copy()

    for hr0, hc0, mask, color in hole_infos:
        _paint_mask(grid, rect_r0 + hr0, rect_c0 + hc0, mask, BLACK)
        _paint_mask(expected, rect_r0 + hr0, rect_c0 + hc0, mask, color)

    occ = np.zeros((height, width), dtype=bool)
    occ[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = True
    sprites: list[tuple[np.ndarray, int]] = []
    for (_hr0, _hc0, mask, color), k in zip(hole_infos, _rotation_plan(hole_count)):
        sprites.append((np.rot90(mask, k).copy(), int(color)))
    random.shuffle(sprites)

    for mask, color in sprites:
        positions = _candidate_locations(occ, mask)
        if not positions:
            return None
        sr0, sc0 = random.choice(positions)
        _paint_mask(grid, sr0, sc0, mask, color)
        occ[sr0 : sr0 + mask.shape[0], sc0 : sc0 + mask.shape[1]] = True

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed385645705 task 4 input")
