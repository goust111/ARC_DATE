from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
OBJECT_COLOR = 3
LINE_COLOR_BY_BENDS = {1: 1, 2: 2, 3: 4, 4: 5}
REQUIRED_BEND_COUNTS = (1, 2, 3, 4, 5)
MIN_GRID_SIZE = 22
MAX_GRID_SIZE = 28
MIN_OBJECT_BORDER = 1
MIN_BOX_GAP = 2
PLACEMENT_TRIES = 1200
GENERATION_TRIES = 500


def _normalize_coords(coords: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _transform_coords(
    coords: tuple[tuple[int, int], ...],
    rotation: int,
    reflect: bool,
) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    for _ in range(rotation % 4):
        pts = [(c, -r) for r, c in pts]
    if reflect:
        pts = [(r, -c) for r, c in pts]
    return _normalize_coords(pts)


def _canonical_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    variants = [_transform_coords(coords, rotation=k, reflect=flip) for k in range(4) for flip in (False, True)]
    return min(variants)


def _count_bends_from_coords(coords: tuple[tuple[int, int], ...]) -> int:
    pts = set(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    total = 0
    for r in range(min(rows), max(rows)):
        for c in range(min(cols), max(cols)):
            filled = 0
            for dr in (0, 1):
                for dc in (0, 1):
                    filled += (r + dr, c + dc) in pts
            if filled == 3:
                total += 1
    return total


def _build_shape_library() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    polyominoes: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 9):
        cur: set[tuple[tuple[int, int], ...]] = set()
        for shape in polyominoes[size - 1]:
            cells = set(shape)
            for r, c in list(cells):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (r + dr, c + dc)
                    if nxt in cells:
                        continue
                    cur.add(_canonical_coords(tuple(sorted(cells | {nxt}))))
        polyominoes[size] = cur

    shapes_by_bends: dict[int, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    for size in range(4, 9):
        for shape in polyominoes[size]:
            bends = _count_bends_from_coords(shape)
            if bends in REQUIRED_BEND_COUNTS:
                shapes_by_bends[bends].append(shape)

    for bends in REQUIRED_BEND_COUNTS:
        if not shapes_by_bends[bends]:
            raise RuntimeError(f"no polyominoes found with bend count {bends}")
    return dict(shapes_by_bends)


SHAPES_BY_BENDS = _build_shape_library()


def _coords_to_mask(coords: tuple[tuple[int, int], ...], color: int = OBJECT_COLOR) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    h = max(rows) + 1
    w = max(cols) + 1
    mask = np.zeros((h, w), dtype=int)
    for r, c in coords:
        mask[r, c] = color
    return mask


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if g[r, c] != OBJECT_COLOR or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == OBJECT_COLOR and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            c0 = min(cols)
            r1 = max(rows)
            c1 = max(cols)
            local = _normalize_coords(coords)
            bends = _count_bends_from_coords(local)
            components.append(
                {
                    "coords": tuple(sorted(coords)),
                    "coord_set": set(coords),
                    "bbox": (r0, c0, r1, c1),
                    "center": (r0 + (r1 - r0) // 2, c0 + (c1 - c0) // 2),
                    "bends": bends,
                    "area": len(coords),
                }
            )
    return components


def _bend_count_from_mask(mask: np.ndarray) -> int:
    total = 0
    for r in range(mask.shape[0] - 1):
        for c in range(mask.shape[1] - 1):
            if int(mask[r : r + 2, c : c + 2].sum()) == 3:
                total += 1
    return total


def _segment_coords(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    coords = []
    step_c = 0 if c1 == c0 else (1 if c1 > c0 else -1)
    cur_c = c0
    while True:
        coords.append((r0, cur_c))
        if cur_c == c1:
            break
        cur_c += step_c

    step_r = 0 if r1 == r0 else (1 if r1 > r0 else -1)
    cur_r = r0 + step_r
    while step_r != 0 and True:
        coords.append((cur_r, c1))
        if cur_r == r1:
            break
        cur_r += step_r
    return coords


def _render_chain(grid: np.ndarray, ordered_objects: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for source, target in zip(ordered_objects, ordered_objects[1:]):
        bends = int(source["bends"])
        if bends not in LINE_COLOR_BY_BENDS:
            raise ValueError(f"source bend count {bends} has no line color mapping")
        color = LINE_COLOR_BY_BENDS[bends]
        for r, c in _segment_coords(source["center"], target["center"]):
            if out[r, c] != OBJECT_COLOR:
                out[r, c] = color
    return out


def _bbox_is_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _pick_shape(bends: int) -> np.ndarray:
    coords = random.choice(SHAPES_BY_BENDS[bends])
    transformed = _transform_coords(coords, rotation=random.randrange(4), reflect=bool(random.getrandbits(1)))
    mask = _coords_to_mask(transformed, color=1)
    if _bend_count_from_mask(mask.astype(bool).astype(int)) != bends:
        raise RuntimeError("shape transform changed bend count unexpectedly")
    return np.where(mask != 0, OBJECT_COLOR, BACKGROUND)


def _instantiate_object(bends: int, mask: np.ndarray, top: int, left: int) -> dict:
    h, w = mask.shape
    bbox = (top, left, top + h - 1, left + w - 1)
    center = (top + (h - 1) // 2, left + (w - 1) // 2)
    coords = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] == OBJECT_COLOR:
                coords.append((top + r, left + c))
    return {
        "bends": bends,
        "mask": mask,
        "bbox": bbox,
        "center": center,
        "coords": tuple(sorted(coords)),
        "coord_set": set(coords),
        "area": int(np.count_nonzero(mask == OBJECT_COLOR)),
    }


def _place_objects(shapes: list[tuple[int, np.ndarray]], grid_shape: tuple[int, int]) -> list[dict] | None:
    grid_h, grid_w = grid_shape
    objects: list[dict] = []
    for bends, mask in shapes:
        h, w = mask.shape
        if h + 2 * MIN_OBJECT_BORDER > grid_h or w + 2 * MIN_OBJECT_BORDER > grid_w:
            return None
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(MIN_OBJECT_BORDER, grid_h - h - MIN_OBJECT_BORDER)
            left = random.randint(MIN_OBJECT_BORDER, grid_w - w - MIN_OBJECT_BORDER)
            candidate = _instantiate_object(bends, mask, top, left)
            center = candidate["center"]
            if any(center[0] == other["center"][0] or center[1] == other["center"][1] for other in objects):
                continue
            if any(not _bbox_is_clear(candidate["bbox"], other["bbox"], MIN_BOX_GAP) for other in objects):
                continue
            objects.append(candidate)
            placed = True
            break
        if not placed:
            return None
    return objects


def _build_input_grid(grid_shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for obj in objects:
        top, left, _, _ = obj["bbox"]
        mask = obj["mask"]
        sub = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
        sub[mask == OBJECT_COLOR] = OBJECT_COLOR
    return grid


def _line_overlap_pixels(ordered_objects: list[dict], green_pixels: set[tuple[int, int]]) -> int:
    seen: dict[tuple[int, int], int] = {}
    overlaps = 0
    for idx, (src, dst) in enumerate(zip(ordered_objects, ordered_objects[1:])):
        for coord in _segment_coords(src["center"], dst["center"]):
            if coord in green_pixels:
                continue
            if coord in seen and seen[coord] != idx:
                overlaps += 1
            seen[coord] = idx
    return overlaps


def _hits_unrelated_object(ordered_objects: list[dict], all_green: set[tuple[int, int]]) -> bool:
    for src, dst in zip(ordered_objects, ordered_objects[1:]):
        allowed = src["coord_set"] | dst["coord_set"]
        for coord in _segment_coords(src["center"], dst["center"]):
            if coord in all_green and coord not in allowed:
                return True
    return False


def _orders_are_nontrivial(objects: list[dict], expected: np.ndarray, grid: np.ndarray) -> bool:
    def safe_render(order: list[dict]) -> np.ndarray | None:
        if any(obj["bends"] not in LINE_COLOR_BY_BENDS for obj in order[:-1]):
            return None
        return _render_chain(grid, order)

    bend_order = sorted(objects, key=lambda obj: obj["bends"])
    row_order = sorted(objects, key=lambda obj: (obj["center"][0], obj["center"][1]))
    col_order = sorted(objects, key=lambda obj: (obj["center"][1], obj["center"][0]))
    area_order = sorted(objects, key=lambda obj: (obj["area"], obj["center"][0], obj["center"][1]))

    if [obj["bends"] for obj in row_order] == [obj["bends"] for obj in bend_order]:
        return False
    if [obj["bends"] for obj in col_order] == [obj["bends"] for obj in bend_order]:
        return False
    if [obj["bends"] for obj in area_order] == [obj["bends"] for obj in bend_order]:
        return False

    row_render = safe_render(row_order)
    col_render = safe_render(col_order)
    area_render = safe_render(area_order)
    if row_render is not None and np.array_equal(expected, row_render):
        return False
    if col_render is not None and np.array_equal(expected, col_render):
        return False
    if area_render is not None and np.array_equal(expected, area_render):
        return False
    return True


def _validate_instance(grid: np.ndarray, overlap_mode: bool, planned_objects: list[dict]) -> bool:
    components = _extract_components(grid)
    if len(components) != len(REQUIRED_BEND_COUNTS):
        return False
    counts = sorted(obj["bends"] for obj in components)
    if counts != list(REQUIRED_BEND_COUNTS):
        return False

    centers = [obj["center"] for obj in components]
    if len({r for r, _ in centers}) != len(centers):
        return False
    if len({c for _, c in centers}) != len(centers):
        return False

    areas = [obj["area"] for obj in components]
    if len(set(areas)) == 1:
        return False

    ordered = sorted(components, key=lambda obj: obj["bends"])
    h_dirs = {int(np.sign(dst["center"][1] - src["center"][1])) for src, dst in zip(ordered, ordered[1:])}
    v_dirs = {int(np.sign(dst["center"][0] - src["center"][0])) for src, dst in zip(ordered, ordered[1:])}
    if len(h_dirs - {0}) < 2 or len(v_dirs - {0}) < 2:
        return False

    all_green = set()
    for obj in components:
        all_green |= obj["coord_set"]
    if _hits_unrelated_object(ordered, all_green):
        return False

    overlap_pixels = _line_overlap_pixels(ordered, all_green)
    if overlap_mode:
        if overlap_pixels <= 0 or overlap_pixels > 6:
            return False
    else:
        if overlap_pixels != 0:
            return False

    expected = _render_chain(grid, ordered)
    if np.array_equal(expected, grid):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if not _orders_are_nontrivial(components, expected, grid):
        return False

    # Independent consistency check against the generator's planned metadata.
    planned_by_bends = {obj["bends"]: obj for obj in planned_objects}
    actual_by_bends = {obj["bends"]: obj for obj in components}
    for bends in REQUIRED_BEND_COUNTS:
        if planned_by_bends[bends]["bbox"] != actual_by_bends[bends]["bbox"]:
            return False
        if planned_by_bends[bends]["center"] != actual_by_bends[bends]["center"]:
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    grid_shape = (random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE), random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE))
    overlap_mode = random.random() < 0.45
    shapes = [(bends, _pick_shape(bends)) for bends in REQUIRED_BEND_COUNTS]
    random.shuffle(shapes)
    objects = _place_objects(shapes, grid_shape)
    if objects is None:
        return None
    grid = _build_input_grid(grid_shape, objects)
    if not _validate_instance(grid, overlap_mode, objects):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort green objects by bend count, then connect adjacent ranks with colored elbow lines.

    Input structure:
    - The grid has a black background with several green 4-connected objects.
    - Each object's topology is described by how many 2x2 windows contain exactly three green pixels.
    - Object centers are the integer centers of their bounding boxes, and these centers do not share rows or columns.

    Rule:
    - Detect every green connected component and count its L-shaped 2x2 bend windows.
    - Sort the objects by bend count from smallest to largest.
    - Connect each object to the next one in that order using an elbow polyline that goes horizontally first, then vertically.
    - Color each segment by the source object's bend count: 1->blue, 2->red, 3->yellow, 4->gray; keep original green pixels on top, and let later segments overwrite earlier ones where lines overlap.

    Key insight:
    - The same geometric count controls both the global connection order and the segment colors, so solving the task requires recovering bend counts rather than using position or size shortcuts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = _extract_components(g)
    if len(objects) < 2:
        return g.copy()
    ordered = sorted(objects, key=lambda obj: obj["bends"])
    return _render_chain(g, ordered)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
