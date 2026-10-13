from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


Coord = tuple[int, int]
Component = dict[str, object]

NEIGHBORS4: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8: tuple[Coord, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
ARC_COLORS = tuple(range(10))

# --------------- solver ---------------


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert order.size > 0, "expected a non-empty grid"
    if order.size > 1:
        assert int(counts[order[0]]) > int(counts[order[1]]), "expected a unique dominant background"
    return int(values[order[0]])


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[Component]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q: deque[Coord] = deque([(r, c)])
            visited[r, c] = True
            cells: list[Coord] = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            components.append({"color": color, "coords": coords, "area": int(coords.shape[0])})

    return components


def _split_anchor_and_object_components(
    components: list[Component],
) -> tuple[dict[int, tuple[Coord, Coord]], list[Component]]:
    by_color: dict[int, list[Component]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    anchor_pairs: dict[int, tuple[Coord, Coord]] = {}
    objects: list[Component] = []

    for color, comps in by_color.items():
        areas = [int(comp["area"]) for comp in comps]
        if any(area == 1 for area in areas):
            assert all(area == 1 for area in areas), "a color cannot mix anchors with regular objects"
            assert len(comps) == 2, "each anchor color must appear as exactly two singleton components"
            coords = [tuple(map(int, comp["coords"][0])) for comp in comps]
            (r0, c0), (r1, c1) = coords
            assert (r0 == r1) ^ (c0 == c1), "anchor pair must align horizontally or vertically"
            anchor_pairs[color] = (coords[0], coords[1])
        else:
            assert all(area >= 3 for area in areas), "regular foreground objects must have area at least three"
            objects.extend(comps)

    assert anchor_pairs, "expected at least one anchor pair"
    return anchor_pairs, objects


def _objects_hit_by_segment(
    pair: tuple[Coord, Coord],
    coord_to_object: dict[Coord, int],
) -> set[int]:
    (r0, c0), (r1, c1) = pair
    touched: set[int] = set()

    if r0 == r1:
        lo, hi = sorted((c0, c1))
        for c in range(lo + 1, hi):
            obj_id = coord_to_object.get((r0, c))
            if obj_id is not None:
                touched.add(obj_id)
    else:
        lo, hi = sorted((r0, r1))
        for r in range(lo + 1, hi):
            obj_id = coord_to_object.get((r, c0))
            if obj_id is not None:
                touched.add(obj_id)

    assert touched, "each anchor segment must cross at least one regular object"
    return touched


def _analyze_scene(
    grid: np.ndarray,
) -> tuple[int, dict[int, tuple[Coord, Coord]], list[Component], dict[int, set[int]]]:
    bg = _dominant_color(grid)
    components = _find_monochrome_components(grid, bg)
    anchor_pairs, objects = _split_anchor_and_object_components(components)

    coord_to_object: dict[Coord, int] = {}
    for obj_id, obj in enumerate(objects):
        coords = np.asarray(obj["coords"], dtype=int)
        for r, c in coords:
            coord_to_object[(int(r), int(c))] = obj_id

    hits_by_color: dict[int, set[int]] = {}
    claimed_objects: dict[int, int] = {}
    for color, pair in anchor_pairs.items():
        touched = _objects_hit_by_segment(pair, coord_to_object)
        hits_by_color[color] = touched
        for obj_id in touched:
            prior = claimed_objects.get(obj_id)
            assert prior in (None, color), "no object may be crossed by multiple anchor segments"
            claimed_objects[obj_id] = color

    return bg, anchor_pairs, objects, hits_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color anchor pair by a virtual straight segment, recolor every crossed
    foreground object to that anchor color, and erase the anchors themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, anchor_pairs, objects, hits_by_color = _analyze_scene(g)
    out = g.copy()

    for pair in anchor_pairs.values():
        for r, c in pair:
            out[r, c] = bg

    for color, touched_ids in hits_by_color.items():
        for obj_id in touched_ids:
            coords = np.asarray(objects[obj_id]["coords"], dtype=int)
            out[coords[:, 0], coords[:, 1]] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
LINE_MARGIN = 2
LINE_GAP = 4
OUTER_PADDING = 1
MAX_PAIRS = 3
MAX_EXTRA_OBJECTS = 2
SHAPE_BOX_MAX = 4
SHAPE_AREA_RANGE = (3, 6)
LINE_OBJECT_TRIES = 160
FREE_OBJECT_TRIES = 200
POSITION_TRIES = 200
GEN_ATTEMPTS = 500


def _normalize_coords(coords: set[Coord] | list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    arr = np.asarray(list(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return tuple((int(r), int(c)) for r, c in arr)


def _transform_shape(shape: tuple[Coord, ...], rot: int, flip: bool) -> tuple[Coord, ...]:
    arr = np.asarray(shape, dtype=int)
    if flip:
        arr[:, 1] *= -1
    for _ in range(rot % 4):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    return _normalize_coords([tuple(map(int, rc)) for rc in arr])


def _shape_dimensions(shape: tuple[Coord, ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _rect_shape(rng: np.random.Generator) -> tuple[Coord, ...]:
    while True:
        h = int(rng.integers(1, 4))
        w = int(rng.integers(1, 5))
        if SHAPE_AREA_RANGE[0] <= h * w <= SHAPE_AREA_RANGE[1]:
            return tuple((r, c) for r in range(h) for c in range(w))


def _l_shape(rng: np.random.Generator) -> tuple[Coord, ...]:
    while True:
        stem = int(rng.integers(2, 5))
        arm = int(rng.integers(2, 5))
        if SHAPE_AREA_RANGE[0] <= stem + arm - 1 <= SHAPE_AREA_RANGE[1]:
            coords = {(r, 0) for r in range(stem)}
            coords.update((stem - 1, c) for c in range(arm))
            return _normalize_coords(coords)


def _t_shape(rng: np.random.Generator) -> tuple[Coord, ...]:
    bar = int(rng.integers(3, 5))
    stem = int(rng.integers(2, 4))
    mid = bar // 2
    coords = {(0, c) for c in range(bar)}
    coords.update((r, mid) for r in range(stem))
    return _normalize_coords(coords)


def _poly_shape(rng: np.random.Generator) -> tuple[Coord, ...]:
    target = int(rng.integers(SHAPE_AREA_RANGE[0], SHAPE_AREA_RANGE[1] + 1))
    cells: set[Coord] = {(0, 0)}

    while len(cells) < target:
        frontier: list[Coord] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                candidate = set(cells)
                candidate.add((nr, nc))
                norm = _normalize_coords(candidate)
                h, w = _shape_dimensions(norm)
                if h <= SHAPE_BOX_MAX and w <= SHAPE_BOX_MAX:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])

    return _normalize_coords(cells)


def _random_shape(rng: np.random.Generator) -> tuple[Coord, ...]:
    builders = (_rect_shape, _l_shape, _t_shape, _poly_shape)
    for _ in range(50):
        shape = builders[int(rng.integers(len(builders)))](rng)
        shape = _transform_shape(shape, int(rng.integers(4)), bool(rng.integers(2)))
        if SHAPE_AREA_RANGE[0] <= len(shape) <= SHAPE_AREA_RANGE[1]:
            return shape
    raise AssertionError("failed to sample a valid object shape")


def _coords_respect_lines(coords: list[Coord], forbidden_rows: set[int], forbidden_cols: set[int]) -> bool:
    return all(r not in forbidden_rows and c not in forbidden_cols for r, c in coords)


def _can_place_cells(grid: np.ndarray, bg: int, coords: list[Coord], pad: int = 1) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != bg:
            return False

    for r, c in coords:
        for nr in range(r - pad, r + pad + 1):
            for nc in range(c - pad, c + pad + 1):
                if (nr, nc) in coord_set:
                    continue
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
                    return False
    return True


def _paint_cells(grid: np.ndarray, coords: list[Coord] | tuple[Coord, ...], color: int) -> None:
    rr = [r for r, _ in coords]
    cc = [c for _, c in coords]
    grid[rr, cc] = color


def _sample_line_counts(rng: np.random.Generator) -> tuple[int, int]:
    total = int(rng.integers(1, MAX_PAIRS + 1))
    if total == 1:
        return (1, 0) if bool(rng.integers(2)) else (0, 1)

    mode = int(rng.integers(3))
    if mode == 0:
        return total, 0
    if mode == 1:
        return 0, total
    if total == 2:
        return 1, 1
    return (2, 1) if bool(rng.integers(2)) else (1, 2)


def _pick_spaced_positions(
    length: int,
    count: int,
    margin: int,
    gap: int,
    rng: np.random.Generator,
) -> list[int] | None:
    if count == 0:
        return []

    candidates = np.arange(margin, length - margin, dtype=int)
    if candidates.size < count:
        return None

    for _ in range(POSITION_TRIES):
        picks = sorted(int(v) for v in rng.choice(candidates, size=count, replace=False))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _allocate_object_budget(n_lines: int, rng: np.random.Generator) -> tuple[list[int], int]:
    max_objects = 9 - n_lines
    touched_counts = [1] * n_lines
    extra_cap = min(MAX_EXTRA_OBJECTS, max_objects - n_lines)
    extra_count = int(rng.integers(0, extra_cap + 1))
    remaining = max_objects - n_lines - extra_count
    bonus = int(rng.integers(0, min(n_lines, remaining) + 1))
    for idx in rng.permutation(n_lines)[:bonus]:
        touched_counts[int(idx)] += 1
    return touched_counts, extra_count


def _place_horizontal_object(
    grid: np.ndarray,
    bg: int,
    line_row: int,
    horizontal_rows: list[int],
    vertical_cols: list[int],
    rng: np.random.Generator,
) -> list[Coord] | None:
    h, w = grid.shape
    forbidden_rows = set(horizontal_rows) - {line_row}
    forbidden_cols = set(vertical_cols)

    for _ in range(LINE_OBJECT_TRIES):
        shape = _random_shape(rng)
        shape_h, shape_w = _shape_dimensions(shape)
        local_rows = sorted({r for r, _ in shape})
        local_row = local_rows[int(rng.integers(len(local_rows)))]
        r0 = line_row - local_row
        if r0 < OUTER_PADDING or r0 + shape_h > h - OUTER_PADDING:
            continue
        max_c0 = w - shape_w - OUTER_PADDING
        if max_c0 < OUTER_PADDING:
            continue
        c0 = int(rng.integers(OUTER_PADDING, max_c0 + 1))
        coords = [(r0 + r, c0 + c) for r, c in shape]
        if not _coords_respect_lines(coords, forbidden_rows, forbidden_cols):
            continue
        line_cols = [c for r, c in coords if r == line_row]
        if min(line_cols) < 2 or max(line_cols) > w - 3:
            continue
        if _can_place_cells(grid, bg, coords):
            return coords
    return None


def _place_vertical_object(
    grid: np.ndarray,
    bg: int,
    line_col: int,
    horizontal_rows: list[int],
    vertical_cols: list[int],
    rng: np.random.Generator,
) -> list[Coord] | None:
    h, w = grid.shape
    forbidden_rows = set(horizontal_rows)
    forbidden_cols = set(vertical_cols) - {line_col}

    for _ in range(LINE_OBJECT_TRIES):
        shape = _random_shape(rng)
        shape_h, shape_w = _shape_dimensions(shape)
        local_cols = sorted({c for _, c in shape})
        local_col = local_cols[int(rng.integers(len(local_cols)))]
        c0 = line_col - local_col
        if c0 < OUTER_PADDING or c0 + shape_w > w - OUTER_PADDING:
            continue
        max_r0 = h - shape_h - OUTER_PADDING
        if max_r0 < OUTER_PADDING:
            continue
        r0 = int(rng.integers(OUTER_PADDING, max_r0 + 1))
        coords = [(r0 + r, c0 + c) for r, c in shape]
        if not _coords_respect_lines(coords, forbidden_rows, forbidden_cols):
            continue
        line_rows = [r for r, c in coords if c == line_col]
        if min(line_rows) < 2 or max(line_rows) > h - 3:
            continue
        if _can_place_cells(grid, bg, coords):
            return coords
    return None


def _place_free_object(
    grid: np.ndarray,
    bg: int,
    horizontal_rows: list[int],
    vertical_cols: list[int],
    rng: np.random.Generator,
) -> list[Coord] | None:
    h, w = grid.shape
    forbidden_rows = set(horizontal_rows)
    forbidden_cols = set(vertical_cols)

    for _ in range(FREE_OBJECT_TRIES):
        shape = _random_shape(rng)
        shape_h, shape_w = _shape_dimensions(shape)
        max_r0 = h - shape_h - OUTER_PADDING
        max_c0 = w - shape_w - OUTER_PADDING
        if max_r0 < OUTER_PADDING or max_c0 < OUTER_PADDING:
            continue
        r0 = int(rng.integers(OUTER_PADDING, max_r0 + 1))
        c0 = int(rng.integers(OUTER_PADDING, max_c0 + 1))
        coords = [(r0 + r, c0 + c) for r, c in shape]
        if not _coords_respect_lines(coords, forbidden_rows, forbidden_cols):
            continue
        if _can_place_cells(grid, bg, coords):
            return coords
    return None


def _is_isolated_background(grid: np.ndarray, bg: int, r: int, c: int) -> bool:
    if int(grid[r, c]) != bg:
        return False
    h, w = grid.shape
    for dr, dc in NEIGHBORS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
            return False
    return True


def _choose_horizontal_anchors(
    grid: np.ndarray,
    bg: int,
    line_row: int,
    object_ids: list[int],
    objects: list[dict[str, object]],
    vertical_cols: list[int],
    rng: np.random.Generator,
) -> tuple[Coord, Coord] | None:
    w = grid.shape[1]
    line_cols = sorted(
        {
            c
            for obj_id in object_ids
            for r, c in objects[obj_id]["coords"]
            if r == line_row
        }
    )
    assert line_cols, "expected horizontal objects to occupy the selector row"
    left_edge = line_cols[0]
    right_edge = line_cols[-1]

    left_candidates = [
        c
        for c in range(0, left_edge - 1)
        if c not in vertical_cols and _is_isolated_background(grid, bg, line_row, c)
    ]
    right_candidates = [
        c
        for c in range(right_edge + 2, w)
        if c not in vertical_cols and _is_isolated_background(grid, bg, line_row, c)
    ]
    if not left_candidates or not right_candidates:
        return None

    c0 = int(rng.choice(left_candidates))
    c1 = int(rng.choice(right_candidates))
    return (line_row, c0), (line_row, c1)


def _choose_vertical_anchors(
    grid: np.ndarray,
    bg: int,
    line_col: int,
    object_ids: list[int],
    objects: list[dict[str, object]],
    horizontal_rows: list[int],
    rng: np.random.Generator,
) -> tuple[Coord, Coord] | None:
    h = grid.shape[0]
    line_rows = sorted(
        {
            r
            for obj_id in object_ids
            for r, c in objects[obj_id]["coords"]
            if c == line_col
        }
    )
    assert line_rows, "expected vertical objects to occupy the selector column"
    top_edge = line_rows[0]
    bottom_edge = line_rows[-1]

    top_candidates = [
        r
        for r in range(0, top_edge - 1)
        if r not in horizontal_rows and _is_isolated_background(grid, bg, r, line_col)
    ]
    bottom_candidates = [
        r
        for r in range(bottom_edge + 2, h)
        if r not in horizontal_rows and _is_isolated_background(grid, bg, r, line_col)
    ]
    if not top_candidates or not bottom_candidates:
        return None

    r0 = int(rng.choice(top_candidates))
    r1 = int(rng.choice(bottom_candidates))
    return (r0, line_col), (r1, line_col)


def _expected_output(
    inp: np.ndarray,
    bg: int,
    objects: list[dict[str, object]],
    lines: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    recolored: set[int] = set()
    for line in lines:
        for r, c in line["anchors"]:
            out[r, c] = bg
        for obj_id in line["object_ids"]:
            assert obj_id not in recolored, "generator must not assign one object to multiple lines"
            recolored.add(int(obj_id))
            coords = np.asarray(objects[int(obj_id)]["coords"], dtype=int)
            out[coords[:, 0], coords[:, 1]] = int(line["anchor_color"])
    return out


def _generated_scene_is_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    n_lines: int,
    n_objects: int,
) -> bool:
    try:
        _bg, anchor_pairs, objects, _hits = _analyze_scene(inp)
    except Exception:
        return False
    if len(anchor_pairs) != n_lines or len(objects) != n_objects:
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    n_h, n_v = _sample_line_counts(rng)
    horizontal_rows = _pick_spaced_positions(h, n_h, LINE_MARGIN, LINE_GAP, rng)
    vertical_cols = _pick_spaced_positions(w, n_v, LINE_MARGIN, LINE_GAP, rng)
    if horizontal_rows is None or vertical_cols is None:
        return None

    n_lines = n_h + n_v
    touched_counts, extra_count = _allocate_object_budget(n_lines, rng)
    total_objects = sum(touched_counts) + extra_count
    colors = [int(c) for c in rng.choice(np.asarray(ARC_COLORS, dtype=int), size=1 + n_lines + total_objects, replace=False)]
    bg = colors[0]
    anchor_colors = colors[1 : 1 + n_lines]
    object_colors = colors[1 + n_lines :]

    grid = np.full((h, w), bg, dtype=int)
    objects: list[dict[str, object]] = []
    lines: list[dict[str, object]] = []

    count_idx = 0
    for row in horizontal_rows:
        lines.append({"orientation": "h", "pos": row, "need": touched_counts[count_idx]})
        count_idx += 1
    for col in vertical_cols:
        lines.append({"orientation": "v", "pos": col, "need": touched_counts[count_idx]})
        count_idx += 1
    rng.shuffle(lines)

    for spec, anchor_color in zip(lines, anchor_colors):
        spec["anchor_color"] = int(anchor_color)
        spec["object_ids"] = []

    next_object_color = 0
    for line in lines:
        for _ in range(int(line["need"])):
            if line["orientation"] == "h":
                coords = _place_horizontal_object(
                    grid,
                    bg,
                    int(line["pos"]),
                    horizontal_rows,
                    vertical_cols,
                    rng,
                )
            else:
                coords = _place_vertical_object(
                    grid,
                    bg,
                    int(line["pos"]),
                    horizontal_rows,
                    vertical_cols,
                    rng,
                )
            if coords is None:
                return None

            color = int(object_colors[next_object_color])
            next_object_color += 1
            _paint_cells(grid, coords, color)
            line["object_ids"].append(len(objects))
            objects.append({"coords": tuple(coords), "color": color})

    for _ in range(extra_count):
        coords = _place_free_object(grid, bg, horizontal_rows, vertical_cols, rng)
        if coords is None:
            return None
        color = int(object_colors[next_object_color])
        next_object_color += 1
        _paint_cells(grid, coords, color)
        objects.append({"coords": tuple(coords), "color": color})

    for line in lines:
        if line["orientation"] == "h":
            anchors = _choose_horizontal_anchors(
                grid,
                bg,
                int(line["pos"]),
                list(line["object_ids"]),
                objects,
                vertical_cols,
                rng,
            )
        else:
            anchors = _choose_vertical_anchors(
                grid,
                bg,
                int(line["pos"]),
                list(line["object_ids"]),
                objects,
                horizontal_rows,
                rng,
            )
        if anchors is None:
            return None
        line["anchors"] = anchors
        _paint_cells(grid, anchors, int(line["anchor_color"]))

    expected = _expected_output(grid, bg, objects, lines)
    if not _generated_scene_is_valid(grid, expected, n_lines=n_lines, n_objects=len(objects)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
