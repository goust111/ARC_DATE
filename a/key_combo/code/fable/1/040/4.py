from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
Direction = tuple[int, int]

BG = 0
WALL = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS: tuple[Direction, ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))
GEN_MAX_ATTEMPTS = 500


@dataclass
class Obj:
    color: int
    box: BBox
    direction: Direction | None = None
    budget: int = 0


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _box_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _box_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _point_in_box(r: int, c: int, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _full_rect(mask: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    area = int(mask.sum())
    return area == _box_h(box) * _box_w(box) and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _parse_component(mask: np.ndarray, color: int) -> Obj:
    coords = np.argwhere(mask)
    assert coords.size, "empty component"
    r0, c0, r1, c1 = _bbox_coords(coords)
    full_box = (r0, c0, r1, c1)
    if _full_rect(mask, full_box):
        assert _box_h(full_box) >= 2 and _box_w(full_box) >= 2, "sliders are solid rectangles"
        return Obj(color=color, box=full_box)

    candidates: list[Obj] = []

    def add_candidate(body: BBox, tail: np.ndarray, direction: Direction, budget: int) -> None:
        if budget < 1 or _box_h(body) < 2 or _box_w(body) < 2:
            return
        br0, bc0, br1, bc1 = body
        if not bool(np.all(mask[br0 : br1 + 1, bc0 : bc1 + 1])):
            return
        expected_area = _box_h(body) * _box_w(body) + budget
        if int(mask.sum()) != expected_area or len(tail) != budget:
            return
        rows = set(map(int, tail[:, 0]))
        cols = set(map(int, tail[:, 1]))
        dr, dc = direction
        if dr == 0:
            if len(rows) == 1:
                candidates.append(Obj(color=color, box=body, direction=direction, budget=budget))
        elif len(cols) == 1:
            candidates.append(Obj(color=color, box=body, direction=direction, budget=budget))

    for bc0 in range(c0 + 1, c1 + 1):
        tail = coords[coords[:, 1] < bc0]
        add_candidate((r0, bc0, r1, c1), tail, (0, 1), bc0 - c0)
    for bc1 in range(c0, c1):
        tail = coords[coords[:, 1] > bc1]
        add_candidate((r0, c0, r1, bc1), tail, (0, -1), c1 - bc1)
    for br0 in range(r0 + 1, r1 + 1):
        tail = coords[coords[:, 0] < br0]
        add_candidate((br0, c0, r1, c1), tail, (1, 0), br0 - r0)
    for br1 in range(r0, r1):
        tail = coords[coords[:, 0] > br1]
        add_candidate((r0, c0, br1, c1), tail, (-1, 0), r1 - br1)

    assert len(candidates) == 1, "each non-rectangular component must have one straight tail"
    return candidates[0]


def _front_cells(box: BBox, direction: Direction) -> list[Coord]:
    r0, c0, r1, c1 = box
    dr, dc = direction
    if dc > 0:
        return [(r, c1 + 1) for r in range(r0, r1 + 1)]
    if dc < 0:
        return [(r, c0 - 1) for r in range(r0, r1 + 1)]
    if dr > 0:
        return [(r1 + 1, c) for c in range(c0, c1 + 1)]
    return [(r0 - 1, c) for c in range(c0, c1 + 1)]


def _shift_box(box: BBox, direction: Direction, steps: int = 1) -> BBox:
    dr, dc = direction
    r0, c0, r1, c1 = box
    return r0 + dr * steps, c0 + dc * steps, r1 + dr * steps, c1 + dc * steps


def _touches(active: set[int], target: int, objects: list[Obj], direction: Direction) -> bool:
    box = objects[target].box
    for idx in active:
        for r, c in _front_cells(objects[idx].box, direction):
            if _point_in_box(r, c, box):
                return True
    return False


def _close_contacts(active: set[int], objects: list[Obj], direction: Direction) -> None:
    changed = True
    while changed:
        changed = False
        for idx in range(len(objects)):
            if idx in active:
                continue
            if _touches(active, idx, objects, direction):
                active.add(idx)
                changed = True


def _can_step(active: set[int], objects: list[Obj], direction: Direction, walls: np.ndarray) -> bool:
    h, w = walls.shape
    for idx in active:
        for r, c in _front_cells(objects[idx].box, direction):
            if not (0 <= r < h and 0 <= c < w):
                return False
            if walls[r, c]:
                return False
            for other, obj in enumerate(objects):
                if other not in active and _point_in_box(r, c, obj.box):
                    return False
    return True


def _simulate_slider(start: int, objects: list[Obj], walls: np.ndarray) -> None:
    obj = objects[start]
    assert obj.direction is not None and obj.budget > 0
    active = {start}
    for _ in range(obj.budget):
        _close_contacts(active, objects, obj.direction)
        if not _can_step(active, objects, obj.direction, walls):
            break
        for idx in sorted(active):
            objects[idx].box = _shift_box(objects[idx].box, obj.direction)


def _parse_objects(grid: np.ndarray, bg: int) -> list[Obj]:
    objects: list[Obj] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {bg, WALL}):
        for coords in _components(grid == color):
            mask = np.zeros_like(grid, dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            objects.append(_parse_component(mask, color))
    assert objects, "expected at least one colored slider"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every tail, then let each tailed solid slider spend its tail length as finite push distance.

    Input structure:
    - The background is black, and gray wall cells touch part of the grid border.
    - Colored objects are filled rectangular sliders; some have a one-cell-wide straight tail attached behind the body.
    - Untailed rectangles may sit on the tailed slider's forward line, and independent slider paths do not cross.

    Rule:
    - Remove all tails and keep gray walls fixed.
    - For each tailed slider, move its rectangular body one cell at a time in the direction away from the tail, for at most the tail length.
    - If its front face touches an untailed rectangle, that rectangle joins the moving chain; contact propagates through already compressed rectangles.
    - Stop a chain as soon as the next step would hit a wall, leave the grid, or exceed the remaining tail budget.

    Key insight:
    - The tail is a finite push budget, so gaps collapse one after another; a front rectangle only starts moving after contact, not for the whole budget.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    walls = g == WALL
    objects = _parse_objects(g, bg)
    slider_indices = [idx for idx, obj in enumerate(objects) if obj.direction is not None]
    assert slider_indices, "expected at least one tailed slider"

    for idx in slider_indices:
        _simulate_slider(idx, objects, walls)

    out = np.full_like(g, bg)
    out[walls] = WALL
    occupied = walls.copy()
    for obj in objects:
        for r, c in _box_cells(obj.box):
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "object moved out of bounds"
            assert not occupied[r, c], "objects and walls must remain disjoint"
            out[r, c] = obj.color
            occupied[r, c] = True
    return out


def _to_actual(direction: Direction, shape: tuple[int, int], cross: int, s: int) -> Coord:
    h, w = shape
    if direction == (0, 1):
        return cross, s
    if direction == (0, -1):
        return cross, w - 1 - s
    if direction == (1, 0):
        return s, cross
    return h - 1 - s, cross


def _forward_box(
    direction: Direction,
    shape: tuple[int, int],
    cross0: int,
    cross_len: int,
    s0: int,
    s1: int,
) -> BBox:
    coords = [_to_actual(direction, shape, cross, s) for cross in range(cross0, cross0 + cross_len) for s in range(s0, s1 + 1)]
    arr = np.array(coords, dtype=int)
    return _bbox_coords(arr)


def _draw_box(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _draw_tail(
    grid: np.ndarray,
    direction: Direction,
    cross: int,
    body_s0: int,
    budget: int,
    color: int,
) -> None:
    shape = grid.shape
    for s in range(body_s0 - budget, body_s0):
        r, c = _to_actual(direction, shape, cross, s)
        assert int(grid[r, c]) == BG
        grid[r, c] = int(color)


def _draw_wall(grid: np.ndarray, direction: Direction, cross0: int, cross_len: int) -> None:
    axis_len = grid.shape[1] if direction[0] == 0 else grid.shape[0]
    for cross in range(cross0, cross0 + cross_len):
        r, c = _to_actual(direction, grid.shape, cross, axis_len - 1)
        grid[r, c] = WALL


def _box_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_offpath_static(grid: np.ndarray, color: int, protected: list[BBox]) -> None:
    h, w = grid.shape
    for _ in range(100):
        bh = random.randint(2, 3)
        bw = random.randint(2, 4)
        r0 = random.randint(1, h - bh - 2)
        c0 = random.randint(1, w - bw - 2)
        box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        if any(_box_overlap(box, other) for other in protected):
            continue
        if np.any(grid[r0 : r0 + bh, c0 : c0 + bw] != BG):
            continue
        _draw_box(grid, box, color)
        return


def _make_chain_scene(branch: str) -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    direction = random.choice(DIRECTIONS)
    axis_len = w if direction[0] == 0 else h
    cross_limit = h if direction[0] == 0 else w
    cross_len = random.randint(2, 3)
    cross0 = random.randint(2, cross_limit - cross_len - 3)
    body_len = random.randint(2, 3)
    budget = random.randint(3, 7)
    body_s0 = random.randint(budget + 1, budget + 4)
    colors = random.sample(COLORS, 6)

    if branch == "free":
        block_count = 0
        gaps: list[int] = []
    elif branch == "budget":
        block_count = random.randint(2, 3)
        gaps = [random.randint(2, 3) for _ in range(block_count)]
        budget = min(budget, random.randint(2, 4))
        body_s0 = random.randint(budget + 1, budget + 4)
    elif branch == "wall":
        block_count = random.randint(1, 3)
        gaps = [random.randint(0, 1) for _ in range(block_count)]
        budget = random.randint(5, 8)
        body_s0 = random.randint(budget + 1, budget + 3)
    else:
        block_count = random.randint(1, 3)
        gaps = [random.randint(0, 2) for _ in range(block_count)]

    intervals: list[tuple[int, int]] = [(body_s0, body_s0 + body_len - 1)]
    cursor = intervals[0][1]
    for gap in gaps:
        start = cursor + gap + 1
        end = start + random.randint(2, 3) - 1
        intervals.append((start, end))
        cursor = end

    if branch == "wall":
        shift = (axis_len - 3) - intervals[-1][1]
        if shift < 0:
            return None
        intervals = [(a + shift, b + shift) for a, b in intervals]
        body_s0 = intervals[0][0]
        if body_s0 - budget < 1:
            return None
        _draw_wall(grid, direction, cross0, cross_len)
    else:
        if intervals[-1][1] + budget >= axis_len - 2:
            return None
        wall_cross = 0 if cross0 > 2 else cross_limit - 2
        _draw_wall(grid, direction, wall_cross, 2)

    protected: list[BBox] = []
    slider_box = _forward_box(direction, grid.shape, cross0, cross_len, intervals[0][0], intervals[0][1])
    _draw_box(grid, slider_box, colors[0])
    tail_cross = random.randint(cross0, cross0 + cross_len - 1)
    _draw_tail(grid, direction, tail_cross, intervals[0][0], budget, colors[0])
    protected.append(slider_box)

    for idx, (s0, s1) in enumerate(intervals[1:], start=1):
        box = _forward_box(direction, grid.shape, cross0, cross_len, s0, s1)
        _draw_box(grid, box, colors[idx])
        protected.append(box)

    if branch in {"free", "budget"} or random.random() < 0.5:
        _place_offpath_static(grid, colors[len(intervals)], protected)

    return grid


def generate_input() -> np.ndarray:
    branches = ("free", "push", "wall", "budget")
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _make_chain_scene(random.choice(branches))
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if grid.shape[0] > 30 or grid.shape[1] > 30 or out.shape[0] > 30 or out.shape[1] > 30:
            continue
        if np.array_equal(grid, out):
            continue
        if not np.any(grid == WALL):
            continue
        return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
