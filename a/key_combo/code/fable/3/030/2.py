from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
AXIS_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
RECT_SIZES = (5, 7, 9)
MAX_HOLE_SIDE = 3
GENERATION_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 80


@dataclass(frozen=True)
class HoleSpec:
    r0: int
    c0: int
    h: int
    w: int

    @property
    def area(self) -> int:
        return self.h * self.w


@dataclass(frozen=True)
class ObjectSpec:
    color: int
    height: int
    width: int
    direction: tuple[int, int]
    hole: HoleSpec
    must_clip: bool


@dataclass(frozen=True)
class PlacedObject:
    spec: ObjectSpec
    r0: int
    c0: int


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _direction_from_offset(dr2: int, dc2: int) -> tuple[int, int]:
    assert (dr2, dc2) != (0, 0), "hole center must be offset from rectangle center"
    if dr2 == 0:
        return 0, _sign(dc2)
    if dc2 == 0:
        return _sign(dr2), 0
    assert abs(dr2) == abs(dc2), "hole offset must be horizontal, vertical, or 45-degree diagonal"
    return _sign(dr2), _sign(dc2)


def _hole_matches_direction(
    rect_h: int,
    rect_w: int,
    hole_r0: int,
    hole_c0: int,
    hole_h: int,
    hole_w: int,
    direction: tuple[int, int],
) -> bool:
    dr, dc = direction
    dr2 = (2 * hole_r0 + hole_h - 1) - (rect_h - 1)
    dc2 = (2 * hole_c0 + hole_w - 1) - (rect_w - 1)
    if (dr2, dc2) == (0, 0):
        return False
    if dr == 0:
        return dr2 == 0 and _sign(dc2) == dc
    if dc == 0:
        return dc2 == 0 and _sign(dr2) == dr
    return _sign(dr2) == dr and _sign(dc2) == dc and abs(dr2) == abs(dc2)


def _valid_holes(rect_h: int, rect_w: int, direction: tuple[int, int]) -> list[HoleSpec]:
    holes: list[HoleSpec] = []
    for h in range(1, min(MAX_HOLE_SIDE, rect_h - 2) + 1):
        for w in range(1, min(MAX_HOLE_SIDE, rect_w - 2) + 1):
            for r0 in range(1, rect_h - h):
                for c0 in range(1, rect_w - w):
                    if _hole_matches_direction(rect_h, rect_w, r0, c0, h, w, direction):
                        holes.append(HoleSpec(r0, c0, h, w))
    return holes


def _ray_cells(
    shape: tuple[int, int],
    r0: int,
    c0: int,
    height: int,
    width: int,
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    grid_h, grid_w = shape
    dr, dc = direction
    center_r = r0 + height // 2
    center_c = c0 + width // 2
    rr, cc = center_r, center_c
    while r0 <= rr < r0 + height and c0 <= cc < c0 + width:
        rr += dr
        cc += dc

    cells: list[tuple[int, int]] = []
    for _ in range(length):
        if not (0 <= rr < grid_h and 0 <= cc < grid_w):
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _rect_bbox(placed: PlacedObject) -> tuple[int, int, int, int]:
    return (
        placed.r0,
        placed.c0,
        placed.r0 + placed.spec.height - 1,
        placed.c0 + placed.spec.width - 1,
    )


def _bboxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 + 1 and br0 <= ar1 + 1 and ac0 <= bc1 + 1 and bc0 <= ac1 + 1


def _cell_in_bbox(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _analyze_object(grid: np.ndarray, color: int) -> dict:
    mask = grid == int(color)
    box = _bbox(mask)
    assert box is not None, "each object color must be present"
    r0, c0, r1, c1 = box
    assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "rectangle center must be a grid cell"

    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((region == BACKGROUND) | (region == int(color))), "rectangle bbox may contain only its color and the hole"
    assert np.all(region[0, :] == int(color)), "hole cannot touch the top edge"
    assert np.all(region[-1, :] == int(color)), "hole cannot touch the bottom edge"
    assert np.all(region[:, 0] == int(color)), "hole cannot touch the left edge"
    assert np.all(region[:, -1] == int(color)), "hole cannot touch the right edge"

    hole_mask = region == BACKGROUND
    hole_box_local = _bbox(hole_mask)
    assert hole_box_local is not None, "each rectangle must contain one black rectangular hole"
    hr0, hc0, hr1, hc1 = hole_box_local
    hole_crop = hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1]
    assert np.all(hole_crop), "hole must be a filled rectangle"
    assert int(hole_mask.sum()) == int(hole_crop.size), "there must be exactly one rectangular hole"

    hole_global = (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1)
    gr0, gc0, gr1, gc1 = hole_global
    dr2 = (gr0 + gr1) - (r0 + r1)
    dc2 = (gc0 + gc1) - (c0 + c1)
    direction = _direction_from_offset(dr2, dc2)
    hole_area = int(hole_mask.sum())

    return {
        "color": int(color),
        "bbox": box,
        "hole_bbox": hole_global,
        "direction": direction,
        "area": hole_area,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangle's hole and fire a same-colored ray encoded by that hole.

    Input structure:
    - The grid has a black background and several separated solid colored rectangles with distinct colors.
    - Each rectangle has exactly one internal black rectangular hole.
    - The hole center is offset from the rectangle center along one of the eight horizontal, vertical, or 45-degree directions.

    Rule:
    - For every rectangle, fill its hole with the rectangle's own color.
    - Use the hole-center offset direction as the ray direction from that rectangle.
    - Starting at the first cell just outside the rectangle on the center row, center column, or center diagonal, draw a same-colored ray whose length equals the hole area, clipping at the grid boundary.

    Key insight:
    - The black hole is not damage to remove; its position gives the arrow direction and its area gives the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    colors = [int(c) for c in np.unique(g) if int(c) != BACKGROUND]
    assert colors, "expected at least one colored rectangle"
    objects = [_analyze_object(g, color) for color in colors]

    out = g.copy()
    for obj in objects:
        color = obj["color"]
        hr0, hc0, hr1, hc1 = obj["hole_bbox"]
        out[hr0 : hr1 + 1, hc0 : hc1 + 1] = color

    planned: dict[tuple[int, int], int] = {}
    for obj in objects:
        color = obj["color"]
        r0, c0, r1, c1 = obj["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        cells = _ray_cells(g.shape, r0, c0, height, width, obj["direction"], obj["area"])
        for cell in cells:
            rr, cc = cell
            previous = planned.get(cell)
            assert previous is None or previous == color, "rays must not conflict"
            assert int(out[rr, cc]) in (BACKGROUND, color), "ray path must be unambiguous"
            planned[cell] = color

    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def _sample_object_spec(color: int, direction: tuple[int, int], must_clip: bool) -> ObjectSpec:
    min_area = 4 if must_clip else 2
    for _ in range(200):
        height = random.choice(RECT_SIZES)
        width = random.choice(RECT_SIZES)
        holes = [h for h in _valid_holes(height, width, direction) if h.area >= min_area]
        if not holes:
            continue
        holes.sort(key=lambda h: (h.area, h.h, h.w))
        hole = random.choice(holes)
        return ObjectSpec(color, height, width, direction, hole, must_clip)
    raise RuntimeError("failed to sample a valid object specification")


def _sample_specs(count: int) -> list[ObjectSpec]:
    colors = random.sample(COLORS, count)
    clipped_direction = random.choice(DIRECTIONS)
    directions = [clipped_direction]
    if clipped_direction not in AXIS_DIRECTIONS:
        directions.append(random.choice(AXIS_DIRECTIONS))
    if clipped_direction not in DIAGONAL_DIRECTIONS:
        directions.append(random.choice(DIAGONAL_DIRECTIONS))
    while len(directions) < count:
        directions.append(random.choice(DIRECTIONS))
    random.shuffle(directions[1:])

    specs = [
        _sample_object_spec(colors[0], directions[0], must_clip=True),
        *[
            _sample_object_spec(colors[i], directions[i], must_clip=False)
            for i in range(1, count)
        ],
    ]
    if len({spec.hole.area for spec in specs}) < 2:
        return _sample_specs(count)
    return specs


def _placement_candidates(
    shape: tuple[int, int],
    spec: ObjectSpec,
    placed: list[PlacedObject],
    occupied_rays: set[tuple[int, int]],
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    grid_h, grid_w = shape
    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []
    for r0 in range(0, grid_h - spec.height + 1):
        for c0 in range(0, grid_w - spec.width + 1):
            candidate = PlacedObject(spec, r0, c0)
            box = _rect_bbox(candidate)
            if any(_bboxes_touch(box, _rect_bbox(other)) for other in placed):
                continue
            if any(_cell_in_bbox(cell, box) for cell in occupied_rays):
                continue

            ray = _ray_cells(shape, r0, c0, spec.height, spec.width, spec.direction, spec.hole.area)
            visible = len(ray)
            if spec.must_clip:
                if not (0 < visible < spec.hole.area):
                    continue
            else:
                if visible != spec.hole.area:
                    continue
            if any(cell in occupied_rays for cell in ray):
                continue
            if any(_cell_in_bbox(cell, _rect_bbox(other)) for other in placed for cell in ray):
                continue
            candidates.append((r0, c0, ray))
    random.shuffle(candidates)
    return candidates


def _place_specs(shape: tuple[int, int], specs: list[ObjectSpec]) -> list[PlacedObject] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        placed: list[PlacedObject] = []
        occupied_rays: set[tuple[int, int]] = set()
        order = [specs[0]] + random.sample(specs[1:], len(specs) - 1)
        failed = False
        for spec in order:
            candidates = _placement_candidates(shape, spec, placed, occupied_rays)
            if not candidates:
                failed = True
                break
            r0, c0, ray = random.choice(candidates[: min(60, len(candidates))])
            placed.append(PlacedObject(spec, r0, c0))
            occupied_rays.update(ray)
        if not failed and _layout_is_clear(shape, placed):
            return placed
    return None


def _layout_is_clear(shape: tuple[int, int], placed: list[PlacedObject]) -> bool:
    rects = [_rect_bbox(obj) for obj in placed]
    ray_owner: dict[tuple[int, int], int] = {}
    clipped = 0
    full = 0
    axis = 0
    diagonal = 0
    for i, obj in enumerate(placed):
        spec = obj.spec
        if spec.direction in AXIS_DIRECTIONS:
            axis += 1
        else:
            diagonal += 1
        ray = _ray_cells(shape, obj.r0, obj.c0, spec.height, spec.width, spec.direction, spec.hole.area)
        if len(ray) < spec.hole.area:
            clipped += 1
        if len(ray) == spec.hole.area:
            full += 1
        if any(_cell_in_bbox(cell, box) for j, box in enumerate(rects) if j != i for cell in ray):
            return False
        for cell in ray:
            owner = ray_owner.get(cell)
            if owner is not None and owner != spec.color:
                return False
            ray_owner[cell] = spec.color
    return clipped >= 1 and full >= 1 and axis >= 1 and diagonal >= 1


def _render_input(shape: tuple[int, int], placed: list[PlacedObject]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in placed:
        spec = obj.spec
        r0, c0 = obj.r0, obj.c0
        grid[r0 : r0 + spec.height, c0 : c0 + spec.width] = spec.color
        hole = spec.hole
        hr0 = r0 + hole.r0
        hc0 = c0 + hole.c0
        grid[hr0 : hr0 + hole.h, hc0 : hc0 + hole.w] = BACKGROUND
    return grid


def _semantic_summary(inp: np.ndarray, out: np.ndarray) -> tuple[set[tuple[int, int]], int, int, int]:
    colors = [int(c) for c in np.unique(inp) if int(c) != BACKGROUND]
    directions: set[tuple[int, int]] = set()
    full = 0
    clipped = 0
    changed = int(np.count_nonzero(inp != out))
    for color in colors:
        obj = _analyze_object(inp, color)
        directions.add(obj["direction"])
        r0, c0, r1, c1 = obj["bbox"]
        ray = _ray_cells(inp.shape, r0, c0, r1 - r0 + 1, c1 - c0 + 1, obj["direction"], obj["area"])
        if len(ray) == obj["area"]:
            full += 1
        if len(ray) < obj["area"]:
            clipped += 1
    return directions, full, clipped, changed


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    count = random.randint(3, 5)
    specs = _sample_specs(count)
    placed = _place_specs((height, width), specs)
    if placed is None:
        return None

    grid = _render_input((height, width), placed)
    try:
        out = main(grid)
        directions, full, clipped, changed = _semantic_summary(grid, out)
    except AssertionError:
        return None
    if max(out.shape) > 30:
        return None
    if not any(d in AXIS_DIRECTIONS for d in directions):
        return None
    if not any(d in DIAGONAL_DIRECTIONS for d in directions):
        return None
    if full < 1 or clipped < 1 or changed < count + 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-vector puzzle input")
