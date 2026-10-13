from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]

BG = 0
ARROW = 7
OBJECT_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(sorted(comp))
    return components


def _arrow_endpoints(cells: list[Coord]) -> tuple[Coord, Coord]:
    assert len(cells) == 3, "each arrow must contain exactly three pixels"
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1:
        row = next(iter(rows))
        cs = sorted(c for _, c in cells)
        assert cs == list(range(cs[0], cs[0] + 3)), "horizontal arrow must be contiguous"
        return (row, cs[0]), (row, cs[2])
    assert len(cols) == 1, "arrow must be horizontal or vertical"
    col = next(iter(cols))
    rs = sorted(r for r, _ in cells)
    assert rs == list(range(rs[0], rs[0] + 3)), "vertical arrow must be contiguous"
    return (rs[0], col), (rs[2], col)


def _adjacent_object_ids(cell: Coord, object_by_cell: dict[Coord, int]) -> set[int]:
    r, c = cell
    hits = set()
    for nr, nc in _neighbors4(r, c):
        obj_id = object_by_cell.get((nr, nc))
        if obj_id is not None:
            hits.add(obj_id)
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each small colored object once in the direction indicated by its adjacent orange arrow.

    Input structure:
    - The background is black.
    - Non-black, non-orange cells form several small 4-connected colored objects.
    - Orange cells form 4-connected three-pixel straight arrows; exactly one endpoint of each arrow is 4-neighbor adjacent to exactly one colored object.

    Rule:
    - For each arrow, use the endpoint touching an object as the tail and the opposite endpoint as the tip, which determines one of the four cardinal directions.
    - Find the unique object pixel with minimum Manhattan distance to the tail.
    - Leave the input unchanged, and stamp a full copy of that object one cell beyond the tip so that the nearest source pixel maps to that anchor cell.

    Key insight:
    - The arrow's facing is not read from geometry alone: the endpoint next to an object must be identified first, and that same attachment also chooses the source pixel that anchors the translated copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    assert bg == BG, "background must be black"

    object_components = _find_components((g != bg) & (g != ARROW))
    assert object_components, "expected at least one colored object"
    for comp in object_components:
        assert 3 <= len(comp) <= 6, "objects must contain between 3 and 6 pixels"

    object_by_cell: dict[Coord, int] = {}
    for obj_id, comp in enumerate(object_components):
        for cell in comp:
            object_by_cell[cell] = obj_id

    arrow_components = _find_components(g == ARROW)
    assert arrow_components, "expected at least one arrow"

    out = g.copy()
    written_copy_cells: set[Coord] = set()
    used_objects: set[int] = set()

    for arrow in arrow_components:
        end_a, end_b = _arrow_endpoints(arrow)
        hits_a = _adjacent_object_ids(end_a, object_by_cell)
        hits_b = _adjacent_object_ids(end_b, object_by_cell)
        is_tail_a = len(hits_a) == 1
        is_tail_b = len(hits_b) == 1
        assert is_tail_a ^ is_tail_b, "exactly one arrow endpoint must touch exactly one object"

        if is_tail_a:
            tail, tip = end_a, end_b
            obj_id = next(iter(hits_a))
            assert not hits_b, "arrow tip must not touch an object"
        else:
            tail, tip = end_b, end_a
            obj_id = next(iter(hits_b))
            assert not hits_a, "arrow tip must not touch an object"

        assert obj_id not in used_objects, "each object may be associated with at most one arrow"
        used_objects.add(obj_id)

        dr = (tip[0] - tail[0]) // 2
        dc = (tip[1] - tail[1]) // 2
        assert (dr, dc) in DIRS
        anchor_dest = (tip[0] + dr, tip[1] + dc)
        assert 0 <= anchor_dest[0] < g.shape[0] and 0 <= anchor_dest[1] < g.shape[1]

        source_cells = object_components[obj_id]
        distances = {
            cell: abs(cell[0] - tail[0]) + abs(cell[1] - tail[1]) for cell in source_cells
        }
        min_dist = min(distances.values())
        anchors = [cell for cell, dist in distances.items() if dist == min_dist]
        assert len(anchors) == 1, "the source anchor pixel must be unique"
        anchor_src = anchors[0]

        for sr, sc in source_cells:
            rr = anchor_dest[0] + (sr - anchor_src[0])
            cc = anchor_dest[1] + (sc - anchor_src[1])
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "copy must stay in bounds"
            assert g[rr, cc] == bg, "copy region must be empty in the input"
            assert (rr, cc) not in written_copy_cells, "copies must not overlap one another"
            written_copy_cells.add((rr, cc))
            out[rr, cc] = g[sr, sc]

    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 24
TOTAL_OBJECTS_MIN = 2
TOTAL_OBJECTS_MAX = 5
ARROWED_OBJECTS_MAX = 3
SHAPE_SIZE_MIN = 3
SHAPE_SIZE_MAX = 6
SHAPE_COORD_BOUND = 3
GEN_ATTEMPTS = 400
GROUP_ATTEMPTS = 120
TRANSLATION_ATTEMPTS = 50


def _normalize_shape(cells: set[Coord]) -> frozenset[Coord]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _random_shape(size: int) -> frozenset[Coord]:
    cells: set[Coord] = {(0, 0)}
    while len(cells) < size:
        frontier: set[Coord] = set()
        for r, c in cells:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                if abs(nr) > SHAPE_COORD_BOUND or abs(nc) > SHAPE_COORD_BOUND:
                    continue
                frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    return _normalize_shape(cells)


def _transform_shape(cells: frozenset[Coord]) -> frozenset[Coord]:
    coords = list(cells)
    if random.random() < 0.5:
        coords = [(r, -c) for r, c in coords]
    for _ in range(random.randrange(4)):
        coords = [(c, -r) for r, c in coords]
    return _normalize_shape(set(coords))


def _translate(cells: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in cells}


def _touches(a: set[Coord], b: set[Coord]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _attachment_candidates(shape: frozenset[Coord], step: Coord) -> list[Coord]:
    dr, dc = step
    shape_set = set(shape)
    candidates: list[Coord] = []
    for r, c in shape:
        arrow_cells = {(r + dr * k, c + dc * k) for k in (1, 2, 3)}
        if shape_set & arrow_cells:
            continue
        tail = (r + dr, c + dc)
        neighbor_count = sum((nr, nc) in shape_set for nr, nc in _neighbors4(*tail))
        if neighbor_count == 1:
            candidates.append((r, c))
    return candidates


def _fits(cells: set[Coord], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _placement_ok(
    shape_cells: set[Coord],
    arrow_cells: set[Coord],
    copy_cells: set[Coord],
    occupied_input: set[Coord],
    reserved_copy: set[Coord],
) -> bool:
    input_cells = shape_cells | arrow_cells
    if input_cells & occupied_input:
        return False
    if input_cells & reserved_copy:
        return False
    if copy_cells & input_cells:
        return False
    if copy_cells & occupied_input:
        return False
    if copy_cells & reserved_copy:
        return False
    if _touches(input_cells, occupied_input):
        return False
    if _touches(input_cells, reserved_copy):
        return False
    if _touches(copy_cells, occupied_input):
        return False
    if _touches(copy_cells, reserved_copy):
        return False
    if _touches(copy_cells, shape_cells):
        return False
    return True


def _try_place_arrowed(
    h: int,
    w: int,
    color: int,
    occupied_input: set[Coord],
    reserved_copy: set[Coord],
) -> dict[str, object] | None:
    for _ in range(GROUP_ATTEMPTS):
        size = random.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        shape = _transform_shape(_random_shape(size))
        step = random.choice(DIRS)
        candidates = _attachment_candidates(shape, step)
        if not candidates:
            continue
        attach = random.choice(candidates)
        dr, dc = step
        shape_local = set(shape)
        arrow_local = {(attach[0] + dr * k, attach[1] + dc * k) for k in (1, 2, 3)}
        copy_local = {(r + 4 * dr, c + 4 * dc) for r, c in shape_local}
        if shape_local & copy_local:
            continue

        all_local = shape_local | arrow_local | copy_local
        min_r = min(r for r, _ in all_local)
        max_r = max(r for r, _ in all_local)
        min_c = min(c for _, c in all_local)
        max_c = max(c for _, c in all_local)
        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            continue

        for _ in range(TRANSLATION_ATTEMPTS):
            off_r = random.randint(row_lo, row_hi)
            off_c = random.randint(col_lo, col_hi)
            shape_cells = _translate(shape_local, off_r, off_c)
            arrow_cells = _translate(arrow_local, off_r, off_c)
            copy_cells = _translate(copy_local, off_r, off_c)
            if not (_fits(shape_cells | arrow_cells | copy_cells, h, w)):
                continue
            if not _placement_ok(shape_cells, arrow_cells, copy_cells, occupied_input, reserved_copy):
                continue
            return {
                "kind": "arrowed",
                "color": color,
                "shape": shape_cells,
                "arrow": arrow_cells,
                "copy": copy_cells,
            }
    return None


def _try_place_plain(
    h: int,
    w: int,
    color: int,
    occupied_input: set[Coord],
    reserved_copy: set[Coord],
) -> dict[str, object] | None:
    for _ in range(GROUP_ATTEMPTS):
        size = random.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        shape = _transform_shape(_random_shape(size))
        shape_local = set(shape)
        max_r = max(r for r, _ in shape_local)
        max_c = max(c for _, c in shape_local)
        for _ in range(TRANSLATION_ATTEMPTS):
            off_r = random.randint(0, h - 1 - max_r)
            off_c = random.randint(0, w - 1 - max_c)
            shape_cells = _translate(shape_local, off_r, off_c)
            if shape_cells & occupied_input:
                continue
            if shape_cells & reserved_copy:
                continue
            if _touches(shape_cells, occupied_input):
                continue
            if _touches(shape_cells, reserved_copy):
                continue
            return {"kind": "plain", "color": color, "shape": shape_cells}
    return None


def _render_grid(h: int, w: int, placements: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((h, w), BG, dtype=int)
    for placement in placements:
        color = int(placement["color"])
        for r, c in placement["shape"]:
            grid[r, c] = color
        if placement["kind"] == "arrowed":
            for r, c in placement["arrow"]:
                grid[r, c] = ARROW

    out = grid.copy()
    for placement in placements:
        if placement["kind"] != "arrowed":
            continue
        color = int(placement["color"])
        for r, c in placement["copy"]:
            out[r, c] = color
    return grid, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    total_objects = random.randint(TOTAL_OBJECTS_MIN, TOTAL_OBJECTS_MAX)
    arrowed_objects = random.randint(1, min(ARROWED_OBJECTS_MAX, total_objects))
    colors = random.sample(OBJECT_COLORS, total_objects)

    placements: list[dict[str, object]] = []
    occupied_input: set[Coord] = set()
    reserved_copy: set[Coord] = set()

    for idx in range(arrowed_objects):
        placement = _try_place_arrowed(h, w, colors[idx], occupied_input, reserved_copy)
        if placement is None:
            return None
        placements.append(placement)
        occupied_input |= set(placement["shape"])
        occupied_input |= set(placement["arrow"])
        reserved_copy |= set(placement["copy"])

    for idx in range(arrowed_objects, total_objects):
        placement = _try_place_plain(h, w, colors[idx], occupied_input, reserved_copy)
        if placement is None:
            return None
        placements.append(placement)
        occupied_input |= set(placement["shape"])

    grid, expected = _render_grid(h, w, placements)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
