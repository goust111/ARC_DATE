from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
MAX_ATTEMPTS = 400


def _shape(rows: list[str]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=int)


SHAPES = [
    _shape(["#", "#"]),
    _shape(["#", "#", "#"]),
    _shape(["#", "#", "#", "#"]),
    _shape(["##"]),
    _shape(["###"]),
    _shape(["####"]),
    _shape(["#.", "##"]),
    _shape([".#", "##"]),
    _shape(["##", "##"]),
    _shape(["###", ".#."]),
    _shape([".#.", "###", ".#."]),
    _shape(["##", "##", "#."]),
    _shape(["##.", ".##"]),
    _shape(["#..", "#..", "###"]),
    _shape(["####", "..#."]),
]

SPINE_SHAPES = [
    shape
    for shape in SHAPES
    if shape.shape[1] in (1, 3) and shape[0, shape.shape[1] // 2] and shape[-1, shape.shape[1] // 2]
]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, color: int = RED) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            objects.append(_describe_object(coords))
    return objects


def _describe_object(coords: np.ndarray) -> dict:
    box = _bbox(coords)
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        rows[int(r)].append(int(c))
        cols[int(c)].append(int(r))
    for values in rows.values():
        values.sort()
    for values in cols.values():
        values.sort()
    r0, c0, r1, c1 = box
    return {
        "coords": coords,
        "bbox": box,
        "rows": dict(rows),
        "cols": dict(cols),
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
        "center_row2": r0 + r1,
        "center_col2": c0 + c1,
    }


def _paint_connector_cell(out: np.ndarray, r: int, c: int) -> None:
    if int(out[r, c]) == RED:
        return
    assert int(out[r, c]) in (BLACK, GREEN)
    out[r, c] = GREEN


def _row_candidate(i: int, a: dict, j: int, b: dict) -> dict | None:
    shared_rows = sorted(set(a["rows"]) & set(b["rows"]))
    usable = []
    for row in shared_rows:
        a_cols = a["rows"][row]
        b_cols = b["rows"][row]
        a_left, a_right = a_cols[0], a_cols[-1]
        b_left, b_right = b_cols[0], b_cols[-1]
        if a_right < b_left:
            usable.append((row, a_right, b_left))
        elif b_right < a_left:
            usable.append((row, b_right, a_left))
    if not usable:
        return None
    row, start, end = usable[len(usable) // 2]
    return {
        "axis": "row",
        "pair": (i, j),
        "score": len(usable),
        "gap": end - start - 1,
        "line": row,
        "start": start,
        "end": end,
    }


def _col_candidate(i: int, a: dict, j: int, b: dict) -> dict | None:
    shared_cols = sorted(set(a["cols"]) & set(b["cols"]))
    usable = []
    for col in shared_cols:
        a_rows = a["cols"][col]
        b_rows = b["cols"][col]
        a_top, a_bottom = a_rows[0], a_rows[-1]
        b_top, b_bottom = b_rows[0], b_rows[-1]
        if a_bottom < b_top:
            usable.append((col, a_bottom, b_top))
        elif b_bottom < a_top:
            usable.append((col, b_bottom, a_top))
    if not usable:
        return None
    col, start, end = usable[len(usable) // 2]
    return {
        "axis": "col",
        "pair": (i, j),
        "score": len(usable),
        "gap": end - start - 1,
        "line": col,
        "start": start,
        "end": end,
    }


def _find_spine(objects: list[dict]) -> dict:
    candidates = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            row_cand = _row_candidate(i, objects[i], j, objects[j])
            if row_cand is not None:
                candidates.append(row_cand)
            col_cand = _col_candidate(i, objects[i], j, objects[j])
            if col_cand is not None:
                candidates.append(col_cand)
    assert candidates, "expected at least one shared-row or shared-column pair after shifting"
    candidates.sort(
        key=lambda cand: (-cand["score"], -cand["gap"], cand["axis"], cand["pair"][0], cand["pair"][1], cand["line"])
    )
    return candidates[0]


def _draw_branch_to_row(out: np.ndarray, obj: dict, spine_row: int, spine_lo: int, spine_hi: int) -> None:
    mid = (spine_lo + spine_hi) / 2
    choices = []
    for r, cols in obj["rows"].items():
        for c in cols:
            if spine_lo <= c <= spine_hi:
                choices.append((abs(r - spine_row), abs(c - mid), r, c))
    assert choices, "branch object must expose a column inside the horizontal spine span"
    _dist, _mid_dist, r, c = min(choices)
    if r < spine_row:
        for rr in range(r + 1, spine_row):
            _paint_connector_cell(out, rr, c)
    else:
        for rr in range(spine_row + 1, r):
            _paint_connector_cell(out, rr, c)


def _draw_branch_to_col(out: np.ndarray, obj: dict, spine_col: int, spine_lo: int, spine_hi: int) -> None:
    mid = (spine_lo + spine_hi) / 2
    choices = []
    for c, rows in obj["cols"].items():
        for r in rows:
            if spine_lo <= r <= spine_hi:
                choices.append((abs(c - spine_col), abs(r - mid), r, c))
    assert choices, "branch object must expose a row inside the vertical spine span"
    _dist, _mid_dist, r, c = min(choices)
    if c < spine_col:
        for cc in range(c + 1, spine_col):
            _paint_connector_cell(out, r, cc)
    else:
        for cc in range(spine_col + 1, c):
            _paint_connector_cell(out, r, cc)


def main(grid: np.ndarray) -> np.ndarray:
    """Shift each red object right by its own width, then connect the shifted layout through the unique spine pair.

    Input structure:
    - A black background contains 4-connected red foreground objects.
    - Each object's horizontal shift amount is its own bounding-box width.
    - The original layout keeps the objects separated; the decisive shared column or row only becomes useful after shifting.

    Rule:
    - Find every red connected component and clear the canvas to black.
    - Repaint each component shifted right by its own bounding-box width.
    - In the shifted layout, find the pair with the longest usable shared row-span or shared column-span and paint the black gap between them green to form the spine.
    - Connect every remaining shifted object orthogonally to that spine with green pixels, while preserving all red object pixels.

    Key insight:
    - The trunk is defined by the post-translation geometry, so the objects must be moved first; only then can the correct spine pair and branch topology be identified.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, RED}, "expected only black background and red objects"

    objects = _find_components(g, color=RED)
    assert 4 <= len(objects) <= 6, "expected 4-6 red objects"

    h, w = g.shape
    out = np.zeros_like(g)
    shifted = []
    occupied = np.zeros((h, w), dtype=bool)

    for obj in objects:
        delta = obj["width"]
        moved = obj["coords"] + np.array([0, delta], dtype=int)
        assert bool((0 <= moved[:, 0]).all() and (moved[:, 0] < h).all())
        assert bool((0 <= moved[:, 1]).all() and (moved[:, 1] < w).all())
        assert not bool(occupied[moved[:, 0], moved[:, 1]].any())
        out[moved[:, 0], moved[:, 1]] = RED
        occupied[moved[:, 0], moved[:, 1]] = True
        shifted.append(_describe_object(moved))

    spine = _find_spine(shifted)
    i, j = spine["pair"]
    if spine["axis"] == "row":
        row = spine["line"]
        for c in range(spine["start"] + 1, spine["end"]):
            _paint_connector_cell(out, row, c)
        for k, obj in enumerate(shifted):
            if k in (i, j):
                continue
            _draw_branch_to_row(out, obj, row, spine["start"], spine["end"])
        return out

    col = spine["line"]
    for r in range(spine["start"] + 1, spine["end"]):
        _paint_connector_cell(out, r, col)
    for k, obj in enumerate(shifted):
        if k in (i, j):
            continue
        _draw_branch_to_col(out, obj, col, spine["start"], spine["end"])
    return out


def _all_centers_unique(objects: list[dict], key: str) -> bool:
    values = [int(obj[key]) for obj in objects]
    return len(values) == len(set(values))


def _duplicated_center_cols(objects: list[dict]) -> list[int]:
    counts = Counter(int(obj["center_col2"]) for obj in objects)
    return [value for value, count in counts.items() if count > 1]


def _shape_local_coords(shape: np.ndarray) -> np.ndarray:
    return np.argwhere(shape > 0).astype(int)


def _shape_global_object(shape: np.ndarray, top: int, left: int) -> dict:
    return _describe_object(_shape_local_coords(shape) + np.array([top, left], dtype=int))


def _distribute_positions(heights: list[int], total_h: int) -> list[int] | None:
    min_needed = sum(heights) + (len(heights) - 1)
    if min_needed > total_h:
        return None
    spare = total_h - min_needed
    slots = [0] * (len(heights) + 1)
    for _ in range(spare):
        slots[random.randrange(len(slots))] += 1
    tops = []
    cursor = slots[0]
    for idx, height in enumerate(heights):
        tops.append(cursor)
        cursor += height
        if idx < len(heights) - 1:
            cursor += 1 + slots[idx + 1]
    return tops


def _feasible_spine_centers(shape_a: np.ndarray, shape_b: np.ndarray, width: int) -> list[int]:
    centers = None
    for shape in (shape_a, shape_b):
        anchor = shape.shape[1] // 2
        lo = shape.shape[1] + anchor
        hi = width - shape.shape[1] + anchor
        vals = set(range(lo, hi + 1))
        centers = vals if centers is None else centers & vals
    if not centers:
        return []
    values = sorted(centers)
    preferred = [c for c in values if 4 <= c <= width - 5]
    return preferred or values


def _input_object_from_output(shape: np.ndarray, top: int, out_left: int) -> dict:
    return _shape_global_object(shape, top, out_left - shape.shape[1])


def _occupied_cols(obj: dict) -> set[int]:
    return {int(col) for col in obj["cols"]}


def _pick_branch_left(
    shape: np.ndarray,
    top: int,
    width: int,
    side: str,
    spine_col: int,
    used_cols: set[int],
    shifted_centers: set[int],
    input_centers: set[int],
) -> int | None:
    coords = _shape_local_coords(shape)
    candidates = []
    for left in range(shape.shape[1], width - shape.shape[1] + 1):
        out_obj = _describe_object(coords + np.array([top, left], dtype=int))
        cols = _occupied_cols(out_obj)
        if cols & used_cols:
            continue
        if side == "left" and max(cols) >= spine_col - 1:
            continue
        if side == "right" and min(cols) <= spine_col + 1:
            continue
        if out_obj["center_col2"] in shifted_centers:
            continue
        inp_obj = _input_object_from_output(shape, top, left)
        if inp_obj["center_col2"] in input_centers:
            continue
        candidates.append(left)
    if not candidates:
        return None
    return random.choice(candidates)


def _build_attempt() -> np.ndarray | None:
    n_objects = random.randint(4, 6)
    branch_count = n_objects - 2

    upper_shape = random.choice(SPINE_SHAPES)
    lower_shape = random.choice(SPINE_SHAPES)
    branch_shapes = [random.choice(SHAPES) for _ in range(branch_count)]
    shapes = [upper_shape, *branch_shapes, lower_shape]
    heights = [shape.shape[0] for shape in shapes]

    min_total_h = sum(heights) + (n_objects - 1)
    if min_total_h > 19:
        return None
    grid_h = random.randint(max(14, min_total_h), 19)
    grid_w = random.randint(15, 22)
    tops = _distribute_positions(heights, grid_h)
    if tops is None:
        return None

    spine_centers = _feasible_spine_centers(upper_shape, lower_shape, grid_w)
    if not spine_centers:
        return None
    spine_col = random.choice(spine_centers)
    upper_left = spine_col - upper_shape.shape[1] // 2
    lower_left = spine_col - lower_shape.shape[1] // 2

    shifted_objects = [
        _shape_global_object(upper_shape, tops[0], upper_left),
        _shape_global_object(lower_shape, tops[-1], lower_left),
    ]
    input_objects = [
        _input_object_from_output(upper_shape, tops[0], upper_left),
        _input_object_from_output(lower_shape, tops[-1], lower_left),
    ]

    if not _all_centers_unique(input_objects, "center_row2"):
        return None
    if not _all_centers_unique(input_objects, "center_col2"):
        return None

    shifted_center_cols = {int(obj["center_col2"]) for obj in shifted_objects}
    input_center_cols = {int(obj["center_col2"]) for obj in input_objects}
    used_cols = _occupied_cols(shifted_objects[0]) | _occupied_cols(shifted_objects[1])

    sides = ["left", "right"]
    while len(sides) < branch_count:
        sides.append(random.choice(["left", "right"]))
    random.shuffle(sides)

    placements = [
        (upper_shape, tops[0], upper_left - upper_shape.shape[1]),
        (lower_shape, tops[-1], lower_left - lower_shape.shape[1]),
    ]

    for shape, top, side in zip(branch_shapes, tops[1:-1], sides):
        left = _pick_branch_left(shape, top, grid_w, side, spine_col, used_cols, shifted_center_cols, input_center_cols)
        if left is None:
            return None
        out_obj = _shape_global_object(shape, top, left)
        inp_obj = _input_object_from_output(shape, top, left)
        shifted_objects.append(out_obj)
        input_objects.append(inp_obj)
        shifted_center_cols.add(int(out_obj["center_col2"]))
        input_center_cols.add(int(inp_obj["center_col2"]))
        used_cols |= _occupied_cols(out_obj)
        placements.append((shape, top, left - shape.shape[1]))

    if len(_duplicated_center_cols(shifted_objects)) != 1:
        return None
    duplicated = _duplicated_center_cols(shifted_objects)[0]
    if sum(int(obj["center_col2"]) == duplicated for obj in shifted_objects) != 2:
        return None
    if not _all_centers_unique(input_objects, "center_col2"):
        return None
    if not _all_centers_unique(input_objects, "center_row2"):
        return None

    width_count = Counter(shape.shape[1] for shape in shapes)
    if len(width_count) < 3:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    for shape, top, left in placements:
        if left < 0 or left + shape.shape[1] > grid_w:
            return None
        inp[top : top + shape.shape[0], left : left + shape.shape[1]][shape > 0] = RED

    objects = _find_components(inp)
    if len(objects) != n_objects:
        return None
    try:
        out = main(inp)
    except AssertionError:
        return None
    if not np.any(out == GREEN):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
