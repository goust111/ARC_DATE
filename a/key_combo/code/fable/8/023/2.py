from __future__ import annotations

import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    RED = 2
    GREEN = 3
    GREY = 5


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, list[list[Coord]]]:
    labels = np.full(mask.shape, -1, dtype=int)
    comps = _component_coords(mask)
    for idx, comp in enumerate(comps):
        for r, c in comp:
            labels[r, c] = idx
    return labels, comps


def _bbox_dims(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _shrink_box(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 - dr, c1 - dc


def _box_center(box: BBox) -> Coord:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _box_contains(outer: BBox, inner: BBox, margin: int = 0) -> bool:
    return (
        outer[0] + margin <= inner[0]
        and outer[1] + margin <= inner[1]
        and inner[2] <= outer[2] - margin
        and inner[3] <= outer[3] - margin
    )


def _boxes_separated(a: BBox, b: BBox, gap: int = 0) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _rect_cycle(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    coords: list[Coord] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _half_arc(box: BBox, start: int) -> list[Coord]:
    cycle = _rect_cycle(box)
    length = len(cycle)
    half = length // 2
    start %= length
    return [cycle[(start + i) % length] for i in range(half)]


def _mirror_coord(coord: Coord, center: Coord) -> Coord:
    r, c = coord
    cr, cc = center
    return 2 * cr - r, 2 * cc - c


def _mirror_set(coords: list[Coord], center: Coord, shape: tuple[int, int]) -> set[Coord] | None:
    mirrored: set[Coord] = set()
    for coord in coords:
        mr, mc = _mirror_coord(coord, center)
        if not in_bounds(shape, mr, mc):
            return None
        mirrored.add((mr, mc))
    return mirrored


def _is_single_closed_loop(loop_cells: set[Coord]) -> bool:
    if len(loop_cells) < 8:
        return False
    start = next(iter(loop_cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        degree = 0
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in loop_cells:
                degree += 1
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        if degree != 2:
            return False
    return len(seen) == len(loop_cells)


def _loop_encloses_center(loop_cells: set[Coord], center: Coord, shape: tuple[int, int]) -> bool:
    cr, cc = center
    if not in_bounds(shape, cr, cc) or (cr, cc) in loop_cells:
        return False
    blocked = set(loop_cells)
    q = deque([(cr, cc)])
    seen = {(cr, cc)}
    touches_boundary = False
    while q:
        r, c = q.popleft()
        if r in {0, shape[0] - 1} or c in {0, shape[1] - 1}:
            touches_boundary = True
            break
        for nr, nc in neighbors4(r, c):
            if not in_bounds(shape, nr, nc):
                touches_boundary = True
                break
            if (nr, nc) in blocked or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return not touches_boundary


def _component_completion(
    comp: list[Coord],
    center: Coord,
    shape: tuple[int, int],
    other_grey: set[Coord],
) -> set[Coord] | None:
    mirrored = _mirror_set(comp, center, shape)
    if mirrored is None:
        return None
    if mirrored & other_grey:
        return None
    completed = set(comp) | mirrored
    if not _is_single_closed_loop(completed):
        return None
    if not _loop_encloses_center(completed, center, shape):
        return None
    return completed


def _assign_loops(grid: np.ndarray) -> tuple[list[Coord], list[set[Coord]]]:
    centers = [tuple(map(int, x)) for x in np.argwhere(grid == Color.RED)]
    grey_mask = grid == Color.GREY
    grey_comps = _component_coords(grey_mask)
    assert 2 <= len(centers) <= 3, "expected two or three red centers"
    assert grey_comps, "expected grey wall fragments"

    all_grey = set(map(tuple, np.argwhere(grey_mask)))
    completed_loops: list[set[Coord]] = []
    for comp in grey_comps:
        comp_set = set(comp)
        valid: list[set[Coord]] = []
        for center in centers:
            completed = _component_completion(comp, center, grid.shape, all_grey - comp_set)
            if completed is not None:
                valid.append(completed)
        assert len(valid) == 1, "each grey fragment must match exactly one center"
        completed_loops.append(valid[0])
    return centers, completed_loops


def _paint_nesting(shape: tuple[int, int], wall_mask: np.ndarray, centers: list[Coord]) -> np.ndarray:
    open_mask = ~wall_mask
    labels, regions = _label_regions(open_mask)
    region_graph = [set() for _ in regions]
    for r, c in np.argwhere(wall_mask):
        ids = {
            int(labels[nr, nc])
            for nr, nc in neighbors4(int(r), int(c))
            if in_bounds(shape, nr, nc) and labels[nr, nc] >= 0
        }
        if len(ids) == 2:
            a, b = sorted(ids)
            region_graph[a].add(b)
            region_graph[b].add(a)

    boundary_ids = {
        int(labels[r, c])
        for r in range(shape[0])
        for c in range(shape[1])
        if open_mask[r, c] and (r in {0, shape[0] - 1} or c in {0, shape[1] - 1})
    }
    assert boundary_ids, "expected at least one exterior region"

    depth = [-1] * len(regions)
    q = deque()
    for rid in sorted(boundary_ids):
        depth[rid] = 0
        q.append(rid)
    while q:
        rid = q.popleft()
        for nbr in region_graph[rid]:
            if depth[nbr] == -1:
                depth[nbr] = depth[rid] + 1
                q.append(nbr)
    assert all(d >= 0 for d in depth), "all open regions must be reachable from the exterior"

    out = np.full(shape, Color.BLACK, dtype=int)
    for rid, region in enumerate(regions):
        color = Color.GREEN if depth[rid] % 2 == 0 else Color.RED
        for r, c in region:
            out[r, c] = color
    out[wall_mask] = Color.GREY
    for r, c in centers:
        out[r, c] = Color.RED
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Assign each grey wall fragment to the unique red center that closes it under 180-degree symmetry, then color regions by nesting parity.

    Input structure:
    - The grid has a black background with grey wall fragments and two or three red single-pixel symmetry centers.
    - Each grey 4-connected fragment is one half of a thin closed wall whose missing half is its 180-degree image around exactly one center.
    - Different completed walls may be disjoint or nested, and a center stays inside the walls assigned to it.

    Rule:
    - For every grey fragment, test each red center by rotating the fragment 180 degrees around that center and unioning the two halves.
    - Keep the unique center whose completion forms a single closed one-pixel-thick loop enclosing that center.
    - After all loops are completed, treat grey loops as walls: color the boundary-connected background green, the next enclosed layer red, then alternate green/red by nesting depth, while preserving the red centers.

    Key insight:
    - Distance is a distractor; the right ownership is the only one whose symmetric completion becomes a valid closed loop, and the region colors are determined only after all loops are restored.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    assert max(grid.shape) <= 30, "ARC grids must fit within 30x30"

    centers, completed_loops = _assign_loops(grid)
    wall_mask = np.zeros(grid.shape, dtype=bool)
    for loop in completed_loops:
        for r, c in loop:
            wall_mask[r, c] = True
    return _paint_nesting(grid.shape, wall_mask, centers)


GEN_MIN_SIZE = 20
GEN_MAX_SIZE = 30
GEN_ATTEMPTS = 600
GEN_ROOT_MARGIN = 1
GEN_SEPARATION = 2


def _sample_odd(lo: int, hi: int) -> int | None:
    vals = [x for x in range(lo, hi + 1) if x % 2 == 1]
    return random.choice(vals) if vals else None


def _sample_box_in(container: BBox, min_h: int, max_h: int, min_w: int, max_w: int) -> BBox | None:
    r0, c0, r1, c1 = container
    max_h = min(max_h, r1 - r0 + 1)
    max_w = min(max_w, c1 - c0 + 1)
    height = _sample_odd(min_h, max_h)
    width = _sample_odd(min_w, max_w)
    if height is None or width is None:
        return None
    start_rows = list(range(r0, r1 - height + 2))
    start_cols = list(range(c0, c1 - width + 2))
    random.shuffle(start_rows)
    random.shuffle(start_cols)
    for sr in start_rows:
        for sc in start_cols:
            box = (sr, sc, sr + height - 1, sc + width - 1)
            if _box_center(box) == ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2):
                return box
    return None


def _sample_disjoint_boxes(container: BBox, count: int, min_h: int, max_h: int, min_w: int, max_w: int) -> list[BBox] | None:
    boxes: list[BBox] = []
    for _ in range(120):
        boxes.clear()
        ok = True
        for _k in range(count):
            placed = None
            for _ in range(120):
                box = _sample_box_in(container, min_h, max_h, min_w, max_w)
                if box is None:
                    break
                if all(_boxes_separated(box, other, GEN_SEPARATION) for other in boxes):
                    placed = box
                    break
            if placed is None:
                ok = False
                break
            boxes.append(placed)
        if ok:
            return boxes.copy()
    return None


def _interior_container(box: BBox) -> BBox:
    return _shrink_box(box, 3, 3)


def _sample_nested_box(parent: BBox, min_h: int, max_h: int, min_w: int, max_w: int) -> BBox | None:
    container = _interior_container(parent)
    if container[0] > container[2] or container[1] > container[3]:
        return None
    return _sample_box_in(container, min_h, max_h, min_w, max_w)


def _choose_inner_loops(group_boxes: list[BBox], parent_of: list[int | None]) -> list[list[BBox]] | None:
    child_map: dict[int, list[BBox]] = {idx: [] for idx in range(len(group_boxes))}
    for child_idx, parent_idx in enumerate(parent_of):
        if parent_idx is not None:
            child_map[parent_idx].append(group_boxes[child_idx])

    loops: list[list[BBox]] = []
    for idx, box in enumerate(group_boxes):
        box_loops = [box]
        h, w = _bbox_dims(box)
        min_dr = min_dc = 2
        max_dr = (h - 5) // 2
        max_dc = (w - 5) // 2
        child_boxes = child_map[idx]
        if child_boxes and random.random() < 0.8:
            candidates: list[tuple[int, int]] = []
            for dr in range(min_dr, max_dr + 1):
                for dc in range(min_dc, max_dc + 1):
                    inner = _shrink_box(box, dr, dc)
                    ih, iw = _bbox_dims(inner)
                    if ih < 5 or iw < 5:
                        continue
                    if all(_box_contains(_shrink_box(inner, 2, 2), child, 0) for child in child_boxes):
                        candidates.append((dr, dc))
            if not candidates:
                return None
            dr, dc = random.choice(candidates)
            box_loops.append(_shrink_box(box, dr, dc))
        elif not child_boxes and max_dr >= min_dr and max_dc >= min_dc and random.random() < 0.55:
            dr = random.randint(min_dr, max_dr)
            dc = random.randint(min_dc, max_dc)
            box_loops.append(_shrink_box(box, dr, dc))
        loops.append(box_loops)
    return loops


def _build_topology(shape: tuple[int, int]) -> tuple[list[BBox], list[int | None]] | None:
    h, w = shape
    full = (GEN_ROOT_MARGIN, GEN_ROOT_MARGIN, h - 1 - GEN_ROOT_MARGIN, w - 1 - GEN_ROOT_MARGIN)
    topology = random.choice(["nested2", "disjoint2", "chain3", "siblings3", "disjoint3"])

    if topology == "nested2":
        outer = _sample_box_in(full, 13, min(25, h - 2), 13, min(25, w - 2))
        if outer is None:
            return None
        inner = _sample_nested_box(outer, 5, 11, 5, 11)
        if inner is None:
            return None
        return [outer, inner], [None, 0]

    if topology == "disjoint2":
        boxes = _sample_disjoint_boxes(full, 2, 5, 11, 5, 11)
        if boxes is None:
            return None
        return boxes, [None, None]

    if topology == "chain3":
        outer = _sample_box_in(full, 17, min(27, h - 2), 17, min(27, w - 2))
        if outer is None:
            return None
        middle = _sample_nested_box(outer, 11, 17, 11, 17)
        if middle is None:
            return None
        inner = _sample_nested_box(middle, 5, 9, 5, 9)
        if inner is None:
            return None
        return [outer, middle, inner], [None, 0, 1]

    if topology == "siblings3":
        outer = _sample_box_in(full, 17, min(27, h - 2), 17, min(27, w - 2))
        if outer is None:
            return None
        inner_container = _interior_container(outer)
        if inner_container[0] > inner_container[2] or inner_container[1] > inner_container[3]:
            return None
        children = _sample_disjoint_boxes(inner_container, 2, 5, 9, 5, 9)
        if children is None:
            return None
        return [outer, children[0], children[1]], [None, 0, 0]

    boxes = _sample_disjoint_boxes(full, 3, 5, 9, 5, 9)
    if boxes is None:
        return None
    return boxes, [None, None, None]


def _input_from_spec(shape: tuple[int, int], centers: list[Coord], loops: list[list[BBox]]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for group_loops in loops:
        for box in group_loops:
            cycle_len = len(_rect_cycle(box))
            start = random.randrange(cycle_len)
            for r, c in _half_arc(box, start):
                grid[r, c] = Color.GREY
    for r, c in centers:
        grid[r, c] = Color.RED
    return grid


def _full_wall_mask(shape: tuple[int, int], loops: list[list[BBox]]) -> np.ndarray:
    wall_mask = np.zeros(shape, dtype=bool)
    for group_loops in loops:
        for box in group_loops:
            for r, c in _rect_cycle(box):
                wall_mask[r, c] = True
    return wall_mask


def _expected_output(shape: tuple[int, int], centers: list[Coord], loops: list[list[BBox]]) -> np.ndarray:
    return _paint_nesting(shape, _full_wall_mask(shape, loops), centers)


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        centers, completed = _assign_loops(inp)
        solved = main(inp)
    except AssertionError:
        return False
    wall_mask = np.zeros(inp.shape, dtype=bool)
    for loop in completed:
        for r, c in loop:
            wall_mask[r, c] = True
    for center in centers:
        if wall_mask[center]:
            return False
    if not np.array_equal(solved, expected):
        return False
    grey_components = _component_coords(inp == Color.GREY)
    if len(grey_components) != len(completed):
        return False
    if len(set(centers)) != len(centers):
        return False
    if np.count_nonzero(inp == Color.GREY) == 0:
        return False
    if not (np.any(expected == Color.GREEN) and np.any(expected == Color.GREY) and np.any(expected == Color.RED)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE),
        random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE),
    )
    built = _build_topology(shape)
    if built is None:
        return None
    group_boxes, parent_of = built
    loops = _choose_inner_loops(group_boxes, parent_of)
    if loops is None:
        return None
    centers = [_box_center(group_loops[0]) for group_loops in loops]
    if len(set(centers)) != len(centers):
        return None

    inp = _input_from_spec(shape, centers, loops)
    expected = _expected_output(shape, centers, loops)
    if not _validate_generated_case(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
