import random
from typing import Iterable

import numpy as np


BG = 0
RECT_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 7, 8)
START_GAP = 3
GEN_RETRIES = 400

SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDE_TOP = "top"
SIDE_BOTTOM = "bottom"
SIDES = (SIDE_LEFT, SIDE_RIGHT, SIDE_TOP, SIDE_BOTTOM)

SYMMETRIC_SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 1), (1, 0), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (1, 1), (2, 1)),
)

ASYMMETRIC_SHAPES = (
    ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 2), (1, 1), (1, 2), (2, 0)),
    ((0, 0), (1, 1), (2, 1), (2, 2)),
)


def _normalize_coords(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(list(coords), dtype=int)
    arr -= arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _shape_mask(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    pts = np.array(list(coords), dtype=int)
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _flip_shape(coords: tuple[tuple[int, int], ...], flip_r: bool, flip_c: bool) -> tuple[tuple[int, int], ...]:
    pts = np.array(coords, dtype=int)
    if flip_r:
        pts[:, 0] = pts[:, 0].max() - pts[:, 0]
    if flip_c:
        pts[:, 1] = pts[:, 1].max() - pts[:, 1]
    return _normalize_coords(map(tuple, pts))


def _is_horizontally_symmetric(coords: Iterable[tuple[int, int]]) -> bool:
    mask = _shape_mask(coords)
    return bool(np.array_equal(mask, np.fliplr(mask)))


for _shape in SYMMETRIC_SHAPES:
    assert _is_horizontally_symmetric(_shape)
for _shape in ASYMMETRIC_SHAPES:
    assert not _is_horizontally_symmetric(_shape)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 8,
) -> list[dict]:
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": _bbox_coords(pts),
                    "area": int(len(coords)),
                }
            )
    return out


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    delta = np.array([dr, dc], dtype=int)
    return coords + delta


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _is_solid_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _classify_side(
    obj_bbox: tuple[int, int, int, int],
    rect_bbox: tuple[int, int, int, int],
) -> str:
    or0, oc0, or1, oc1 = obj_bbox
    rr0, rc0, rr1, rc1 = rect_bbox
    candidates = []
    row_overlap = not (or1 < rr0 or rr1 < or0)
    col_overlap = not (oc1 < rc0 or rc1 < oc0)
    if oc1 < rc0 and row_overlap:
        candidates.append(SIDE_LEFT)
    if oc0 > rc1 and row_overlap:
        candidates.append(SIDE_RIGHT)
    if or1 < rr0 and col_overlap:
        candidates.append(SIDE_TOP)
    if or0 > rr1 and col_overlap:
        candidates.append(SIDE_BOTTOM)
    assert len(candidates) == 1, f"expected one alignment axis, got {candidates}"
    return candidates[0]


def _target_shift(
    side: str,
    symmetric: bool,
    obj_bbox: tuple[int, int, int, int],
    rect_bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    or0, oc0, or1, oc1 = obj_bbox
    rr0, rc0, rr1, rc1 = rect_bbox
    h, w = grid_shape
    if side == SIDE_LEFT:
        target_c0 = rc0 - 2 - (oc1 - oc0) if symmetric else 0
        return 0, target_c0 - oc0
    if side == SIDE_RIGHT:
        target_c0 = rc1 + 2 if symmetric else w - (oc1 - oc0 + 1)
        return 0, target_c0 - oc0
    if side == SIDE_TOP:
        target_r0 = rr0 - 2 - (or1 - or0) if symmetric else 0
        return target_r0 - or0, 0
    assert side == SIDE_BOTTOM
    target_r0 = rr1 + 2 if symmetric else h - (or1 - or0 + 1)
    return target_r0 - or0, 0


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _find_components(g, bg=bg, connectivity=8)

    rects = [comp for comp in comps if comp["color"] == RECT_COLOR and _is_solid_rectangle(comp)]
    assert len(rects) == 1, "expected one solid gray rectangle"
    rect = rects[0]
    rect_bbox = rect["bbox"]

    objects = [comp for comp in comps if comp is not rect]
    assert 4 <= len(objects) <= 6, "expected 4 to 6 foreground objects"

    out = np.full_like(g, bg)
    _paint(out, rect["coords"], RECT_COLOR)
    occupied = {tuple(cell) for cell in rect["coords"]}

    for comp in sorted(objects, key=lambda item: int(item["color"])):
        color = int(comp["color"])
        assert color in OBJECT_COLORS
        bbox = comp["bbox"]
        side = _classify_side(bbox, rect_bbox)
        local = comp["coords"] - np.array([bbox[0], bbox[1]], dtype=int)
        symmetric = _is_horizontally_symmetric(local)
        dr, dc = _target_shift(side, symmetric, bbox, rect_bbox, g.shape)
        moved = _translate(comp["coords"], dr, dc)
        moved_bbox = _bbox_coords(moved)
        mr0, mc0, mr1, mc1 = moved_bbox
        assert 0 <= mr0 <= mr1 < g.shape[0]
        assert 0 <= mc0 <= mc1 < g.shape[1]
        for cell in map(tuple, moved):
            assert cell not in occupied, "moved objects must stay disjoint"
        _paint(out, moved, color)
        occupied.update(map(tuple, moved))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored object either toward the gray rectangle or away to the edge based on horizontal mirror symmetry.

    Input structure:
    - The grid has black background, one solid gray rectangle near the center, and 4 to 6 small single-color 8-connected objects around it.
    - Every small object is uniquely aligned with the rectangle on exactly one axis: either left/right with overlapping rows, or top/bottom with overlapping columns.
    - Each small object's tight-bbox foreground mask is either horizontally mirror-symmetric or not.

    Rule:
    - Find each non-gray foreground object and test whether its tight-bbox mask matches its left-right reflection.
    - If the object is symmetric, slide it along its alignment axis toward the gray rectangle until exactly one background cell remains between the object and the rectangle.
    - If the object is not symmetric, slide it along the same axis away from the gray rectangle until the object touches the outer grid boundary.
    - Clear the old object positions and keep the gray rectangle unchanged.

    Key insight:
    - The geometry of each object, not its color or side, determines the motion direction: horizontally symmetric shapes are attracted to the rectangle, asymmetric shapes are repelled to the nearest edge.
    """
    return _transform(grid)


def _choose_shape(symmetric: bool) -> tuple[tuple[int, int], ...]:
    base = random.choice(SYMMETRIC_SHAPES if symmetric else ASYMMETRIC_SHAPES)
    return _flip_shape(base, flip_r=bool(random.getrandbits(1)), flip_c=bool(random.getrandbits(1)))


def _make_side_counts(count: int) -> dict[str, int]:
    counts = {side: 1 for side in SIDES}
    for _ in range(count - 4):
        choices = [side for side in SIDES if counts[side] < 2]
        counts[random.choice(choices)] += 1
    return counts


def _assign_symmetry_labels(sides: list[str]) -> list[bool]:
    labels = [None] * len(sides)
    horizontal = [i for i, side in enumerate(sides) if side in (SIDE_LEFT, SIDE_RIGHT)]
    vertical = [i for i, side in enumerate(sides) if side in (SIDE_TOP, SIDE_BOTTOM)]
    for axis_indices in (horizontal, vertical):
        sym_i, asym_i = random.sample(axis_indices, 2)
        labels[sym_i] = True
        labels[asym_i] = False
    for i in range(len(labels)):
        if labels[i] is None:
            labels[i] = bool(random.getrandbits(1))
    return [bool(v) for v in labels]


def _choose_centers(
    side: str,
    size: int,
    rect_bbox: tuple[int, int, int, int],
    used: dict[str, list[tuple[int, int]]],
) -> tuple[int, tuple[int, int]] | None:
    rr0, rc0, rr1, rc1 = rect_bbox
    half = size // 2
    if side in (SIDE_LEFT, SIDE_RIGHT):
        centers = list(range(rr0, rr1 + 1))
    else:
        centers = list(range(rc0, rc1 + 1))
    random.shuffle(centers)
    for center in centers:
        interval = (center - half, center + half)
        if all(not _intervals_overlap(interval, other) for other in used[side]):
            return center, interval
    return None


def _place_object(
    grid_shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    side: str,
    color: int,
    symmetric: bool,
    used_intervals: dict[str, list[tuple[int, int]]],
) -> dict | None:
    rr0, rc0, rr1, rc1 = rect_bbox
    shape = _choose_shape(symmetric)
    pts = np.array(shape, dtype=int)
    shape_h = int(pts[:, 0].max()) + 1
    shape_w = int(pts[:, 1].max()) + 1

    center_pick = _choose_centers(
        side,
        shape_h if side in (SIDE_LEFT, SIDE_RIGHT) else shape_w,
        rect_bbox,
        used_intervals,
    )
    if center_pick is None:
        return None
    center, interval = center_pick

    if side == SIDE_LEFT:
        top = center - shape_h // 2
        left = rc0 - START_GAP - shape_w
    elif side == SIDE_RIGHT:
        top = center - shape_h // 2
        left = rc1 + START_GAP + 1
    elif side == SIDE_TOP:
        top = rr0 - START_GAP - shape_h
        left = center - shape_w // 2
    else:
        top = rr1 + START_GAP + 1
        left = center - shape_w // 2

    bottom = top + shape_h - 1
    right = left + shape_w - 1
    h, w = grid_shape
    if not (0 <= top and 0 <= left and bottom < h and right < w):
        return None

    abs_coords = pts + np.array([top, left], dtype=int)
    used_intervals[side].append(interval)
    return {
        "color": int(color),
        "side": side,
        "symmetric": bool(symmetric),
        "coords": abs_coords,
        "bbox": _bbox_coords(abs_coords),
    }


def _expected_from_specs(
    grid_shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    objects: list[dict],
) -> np.ndarray:
    h, w = grid_shape
    out = np.full((h, w), BG, dtype=int)
    rr0, rc0, rr1, rc1 = rect_bbox
    out[rr0 : rr1 + 1, rc0 : rc1 + 1] = RECT_COLOR
    for spec in objects:
        dr, dc = _target_shift(spec["side"], spec["symmetric"], spec["bbox"], rect_bbox, grid_shape)
        moved = _translate(spec["coords"], dr, dc)
        _paint(out, moved, spec["color"])
    return out


def _validate_paths(grid: np.ndarray, rect_bbox: tuple[int, int, int, int], objects: list[dict]) -> None:
    rr0, rc0, rr1, rc1 = rect_bbox
    rect_cells = {(r, c) for r in range(rr0, rr1 + 1) for c in range(rc0, rc1 + 1)}
    all_object_cells = [set(map(tuple, spec["coords"])) for spec in objects]

    for i, spec in enumerate(objects):
        static = set(rect_cells)
        for j, other_cells in enumerate(all_object_cells):
            if i != j:
                static.update(other_cells)
        dr, dc = _target_shift(spec["side"], spec["symmetric"], spec["bbox"], rect_bbox, grid.shape)
        steps = max(abs(dr), abs(dc))
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
        current = spec["coords"].copy()
        for _ in range(steps):
            current = _translate(current, step_r, step_c)
            assert not any(tuple(cell) in static for cell in current), "slide path intersects another object or rectangle"


def _validate_instance(grid: np.ndarray, rect_bbox: tuple[int, int, int, int], specs: list[dict]) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 4 <= len(specs) <= 6
    assert max(grid.shape) <= 30
    assert {spec["side"] for spec in specs} == set(SIDES) or len(specs) > 4

    horizontal = [spec for spec in specs if spec["side"] in (SIDE_LEFT, SIDE_RIGHT)]
    vertical = [spec for spec in specs if spec["side"] in (SIDE_TOP, SIDE_BOTTOM)]
    assert any(spec["symmetric"] for spec in horizontal)
    assert any(not spec["symmetric"] for spec in horizontal)
    assert any(spec["symmetric"] for spec in vertical)
    assert any(not spec["symmetric"] for spec in vertical)

    color_set = {spec["color"] for spec in specs}
    assert len(color_set) == len(specs)
    assert color_set.issubset(set(OBJECT_COLORS))

    comps = _find_components(grid, bg=BG, connectivity=8)
    assert len(comps) == len(specs) + 1

    occupied = set()
    for spec in specs:
        assert 3 <= len(spec["coords"]) <= 5
        local = spec["coords"] - np.array([spec["bbox"][0], spec["bbox"][1]], dtype=int)
        assert _is_horizontally_symmetric(local) == spec["symmetric"]
        for cell in map(tuple, spec["coords"]):
            assert cell not in occupied
            occupied.add(cell)

    _validate_paths(grid, rect_bbox, specs)

    expected = _expected_from_specs(grid.shape, rect_bbox, specs)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver must match the constructed semantics"

    moved_sets = []
    for spec in specs:
        dr, dc = _target_shift(spec["side"], spec["symmetric"], spec["bbox"], rect_bbox, grid.shape)
        assert dr != 0 or dc != 0, "every object should move"
        moved = _translate(spec["coords"], dr, dc)
        moved_sets.append(set(map(tuple, moved)))
        if spec["symmetric"]:
            if spec["side"] == SIDE_LEFT:
                assert moved[:, 1].max() == rect_bbox[1] - 2
            elif spec["side"] == SIDE_RIGHT:
                assert moved[:, 1].min() == rect_bbox[3] + 2
            elif spec["side"] == SIDE_TOP:
                assert moved[:, 0].max() == rect_bbox[0] - 2
            else:
                assert moved[:, 0].min() == rect_bbox[2] + 2
        else:
            if spec["side"] == SIDE_LEFT:
                assert moved[:, 1].min() == 0
            elif spec["side"] == SIDE_RIGHT:
                assert moved[:, 1].max() == grid.shape[1] - 1
            elif spec["side"] == SIDE_TOP:
                assert moved[:, 0].min() == 0
            else:
                assert moved[:, 0].max() == grid.shape[0] - 1

    rect_cells = {(r, c) for r in range(rect_bbox[0], rect_bbox[2] + 1) for c in range(rect_bbox[1], rect_bbox[3] + 1)}
    seen = set(rect_cells)
    for moved in moved_sets:
        assert moved.isdisjoint(rect_cells)
        assert moved.isdisjoint(seen - rect_cells)
        seen.update(moved)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    rect_h = random.randint(4, 5)
    rect_w = random.randint(5, 6)

    row_center = h // 2 - rect_h // 2
    col_center = w // 2 - rect_w // 2
    rr0_lo = max(7, row_center - 2)
    rr0_hi = min(h - rect_h - 7, row_center + 2)
    rc0_lo = max(7, col_center - 2)
    rc0_hi = min(w - rect_w - 7, col_center + 2)
    if rr0_lo > rr0_hi or rc0_lo > rc0_hi:
        return None

    rr0 = random.randint(rr0_lo, rr0_hi)
    rc0 = random.randint(rc0_lo, rc0_hi)
    rect_bbox = (rr0, rc0, rr0 + rect_h - 1, rc0 + rect_w - 1)

    obj_count = random.randint(4, 6)
    side_counts = _make_side_counts(obj_count)
    side_list = []
    for side in SIDES:
        side_list.extend([side] * side_counts[side])
    random.shuffle(side_list)
    symmetric_flags = _assign_symmetry_labels(side_list)
    colors = random.sample(OBJECT_COLORS, obj_count)

    used_intervals = {side: [] for side in SIDES}
    specs = []
    for side, symmetric, color in zip(side_list, symmetric_flags, colors):
        spec = _place_object((h, w), rect_bbox, side, color, symmetric, used_intervals)
        if spec is None:
            return None
        specs.append(spec)

    grid = np.full((h, w), BG, dtype=int)
    rr0, rc0, rr1, rc1 = rect_bbox
    grid[rr0 : rr1 + 1, rc0 : rc1 + 1] = RECT_COLOR
    for spec in specs:
        _paint(grid, spec["coords"], spec["color"])

    _validate_instance(grid, rect_bbox, specs)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
