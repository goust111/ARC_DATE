from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
RAY_COLOR = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

FRAME_SIZE_CHOICES = (5, 7)
GRID_SIDE_RANGE = (20, 26)
OUTSIDE_OBJECT_COUNT_RANGE = (4, 6)
TEMPLATE_SIZE_RANGE = (3, 6)
DISTRACTOR_SIZE_RANGE = (3, 8)
GEN_ATTEMPTS = 600
CORNER_MARGIN_RANGE = (3, 6)


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1 + 1) // 2, (c0 + c1 + 1) // 2


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return (
        max(r for r, _c in shape) + 1,
        max(c for _r, c in shape) + 1,
    )


def _shape_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _abs_coords(shape: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _inside_bbox_cell(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _all_inside_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.all((r0 <= rows) & (rows <= r1) & (c0 <= cols) & (cols <= c1)))


def _find_components(grid: np.ndarray, *, excluded_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in excluded_colors or visited[r, c]:
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_connected_shape(shape_cells: tuple[tuple[int, int], ...]) -> bool:
    cells = set(shape_cells)
    seen = {next(iter(cells))}
    q = deque(seen)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _build_shape_pool() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(4) for c in range(4)]
    pool: dict[int, list[tuple[tuple[int, int], ...]]] = {size: [] for size in range(3, 9)}
    for size in range(3, 9):
        seen = set()
        for combo in itertools.combinations(cells, size):
            if not _is_connected_shape(combo):
                continue
            key = _shape_key(np.array(combo, dtype=int))
            if key in seen:
                continue
            seen.add(key)
            pool[size].append(key)
    return pool


SHAPES_BY_SIZE = _build_shape_pool()
ALL_SHAPES = [shape for size in range(3, 9) for shape in SHAPES_BY_SIZE[size]]


def _is_l_frame(comp: dict) -> bool:
    if int(comp["color"]) != RAY_COLOR:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if r1 <= r0 or c1 <= c0:
        return False
    expected = {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    actual = {tuple(map(int, rc)) for rc in comp["coords"]}
    return actual == expected


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g, excluded_colors={BG})
    frame_candidates = [comp for comp in comps if _is_l_frame(comp)]
    assert len(frame_candidates) == 1, "expected exactly one grey L-frame"
    frame = frame_candidates[0]
    fr0, fc0, fr1, fc1 = frame["bbox"]
    inner_bbox = (fr0, fc0, fr1 - 1, fc1 - 1)

    template_candidates = [
        comp
        for comp in comps
        if int(comp["color"]) != RAY_COLOR and _all_inside_bbox(comp["coords"], inner_bbox)
    ]
    assert len(template_candidates) == 1, "expected exactly one template inside the frame"
    template = template_candidates[0]
    template_shape = _shape_key(template["coords"])

    outside = []
    for comp in comps:
        if comp is frame or comp is template:
            continue
        coords = comp["coords"]
        assert not np.any(
            (fr0 <= coords[:, 0]) & (coords[:, 0] <= fr1) & (fc0 <= coords[:, 1]) & (coords[:, 1] <= fc1)
        ), "outside objects must not overlap the frame bbox"
        outside.append(comp)

    assert OUTSIDE_OBJECT_COUNT_RANGE[0] <= len(outside) <= OUTSIDE_OBJECT_COUNT_RANGE[1]

    matches = [comp for comp in outside if _shape_key(comp["coords"]) == template_shape]
    assert len(matches) == 1, "expected exactly one outside shape match"
    match = matches[0]
    distractors = [comp for comp in outside if comp is not match]
    return {
        "frame": frame,
        "template": template,
        "template_shape": template_shape,
        "outside": outside,
        "match": match,
        "distractors": distractors,
    }


def _distance_key(obj_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> int:
    ro, co = _center_from_bbox(obj_bbox)
    rf, cf = _center_from_bbox(frame_bbox)
    return abs(ro - rf) + abs(co - cf)


def _ray_cells(coords: np.ndarray, distance: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for k in range(1, distance + 1):
            for nr, nc in ((rr - k, cc), (rr + k, cc), (rr, cc - k), (rr, cc + k)):
                if 0 <= nr < h and 0 <= nc < w:
                    cells.add((nr, nc))
    return cells


def _safe_match_position(
    coords: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
) -> tuple[bool, int]:
    obj_bbox = _bbox_from_coords(coords)
    distance = _distance_key(obj_bbox, frame_bbox)
    if distance < 2:
        return False, distance
    h, w = grid_shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr - distance < 0 or rr + distance >= h or cc - distance < 0 or cc + distance >= w:
            return False, distance
        for k in range(1, distance + 1):
            if (
                _inside_bbox_cell(rr - k, cc, frame_bbox)
                or _inside_bbox_cell(rr + k, cc, frame_bbox)
                or _inside_bbox_cell(rr, cc - k, frame_bbox)
                or _inside_bbox_cell(rr, cc + k, frame_bbox)
            ):
                return False, distance
    return True, distance


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the framed template, find the uniquely matching outside object, and extend grey rays from it with frame-distance length.

    Input structure:
    - The background is black.
    - Exactly one grey L-frame appears in a corner-positioned bbox; its horizontal leg is the bottom row and its vertical leg is the right column.
    - A single colored template object lies inside the frame's upper-left interior, while 4-6 other colored 4-connected objects are scattered elsewhere.

    Rule:
    - Extract the template's normalized binary mask from inside the grey frame.
    - Among the outside objects, find the unique object with the same normalized mask, ignoring color.
    - Erase every other outside object to black.
    - Compute the Manhattan distance between the bbox center of the matched object and the bbox center of the L-frame.
    - From every pixel of the matched object, draw grey rays upward, downward, leftward, and rightward for exactly that many cells, while keeping the matched object, the frame, and the framed template unchanged.

    Key insight:
    - The same L-frame encodes both parameters: its interior template selects which object is active, and its spatial distance to that object sets the ray length.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    frame = scene["frame"]
    template = scene["template"]
    match = scene["match"]
    distractors = scene["distractors"]

    out = g.copy()
    for comp in distractors:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    protected = set()
    for comp in (frame, template, match):
        protected |= {tuple(map(int, rc)) for rc in comp["coords"]}

    distance = _distance_key(match["bbox"], frame["bbox"])
    for r, c in match["coords"]:
        rr = int(r)
        cc = int(c)
        for k in range(1, distance + 1):
            for nr, nc in ((rr - k, cc), (rr + k, cc), (rr, cc - k), (rr, cc + k)):
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and (nr, nc) not in protected:
                    out[nr, nc] = RAY_COLOR

    return out


def _frame_bbox(corner: str, side: int, frame_size: int) -> tuple[int, int, int, int]:
    top_margin = random.randint(*CORNER_MARGIN_RANGE)
    left_margin = random.randint(*CORNER_MARGIN_RANGE)
    if corner == "tl":
        r0, c0 = top_margin, left_margin
    elif corner == "tr":
        r0, c0 = top_margin, side - frame_size - left_margin
    elif corner == "bl":
        r0, c0 = side - frame_size - top_margin, left_margin
    else:
        r0, c0 = side - frame_size - top_margin, side - frame_size - left_margin
    return r0, c0, r0 + frame_size - 1, c0 + frame_size - 1


def _paint_frame_and_template(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    template_shape: tuple[tuple[int, int], ...],
    template_color: int,
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    grid[r1, c0 : c1 + 1] = RAY_COLOR
    grid[r0 : r1 + 1, c1] = RAY_COLOR

    h, w = _shape_dims(template_shape)
    inner_size = r1 - r0
    top = r0 + random.randint(0, inner_size - h)
    left = c0 + random.randint(0, inner_size - w)
    coords = _abs_coords(template_shape, top, left)
    grid[coords[:, 0], coords[:, 1]] = int(template_color)
    return coords


def _cells_clear(coords: np.ndarray, occupied: np.ndarray, gap: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        for nr in range(max(0, rr - gap), min(h, rr + gap + 1)):
            for nc in range(max(0, cc - gap), min(w, cc + gap + 1)):
                if occupied[nr, nc]:
                    return False
    return True


def _candidate_positions(
    shape: tuple[tuple[int, int], ...],
    occupied: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    h, w = occupied.shape
    sh, sw = _shape_dims(shape)
    out = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            coords = _abs_coords(shape, top, left)
            if _cells_clear(coords, occupied, gap=1):
                out.append((top, left, coords))
    random.shuffle(out)
    return out


def _choose_distractor_shapes(
    template_shape: tuple[tuple[int, int], ...],
    distractor_count: int,
) -> list[tuple[tuple[int, int], ...]]:
    template_area = len(template_shape)
    template_dims = _shape_dims(template_shape)
    chosen: list[tuple[tuple[int, int], ...]] = []
    used = {template_shape}

    same_area = [shape for shape in SHAPES_BY_SIZE[template_area] if shape not in used]
    same_bbox = [shape for shape in ALL_SHAPES if _shape_dims(shape) == template_dims and shape not in used]
    random.shuffle(same_area)
    random.shuffle(same_bbox)

    for bucket in (same_area, same_bbox):
        for shape in bucket:
            if shape not in used:
                chosen.append(shape)
                used.add(shape)
                break

    remaining_pool = [shape for shape in ALL_SHAPES if 3 <= len(shape) <= 8 and shape not in used]
    random.shuffle(remaining_pool)
    for shape in remaining_pool:
        if len(chosen) >= distractor_count:
            break
        chosen.append(shape)
        used.add(shape)

    assert len(chosen) >= distractor_count
    return chosen[:distractor_count]


def _place_role_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    remaining: list[tuple[tuple[tuple[int, int], ...], int]],
    predicate,
) -> tuple[bool, dict | None]:
    options = remaining[:]
    random.shuffle(options)
    for shape, color in options:
        positions = _candidate_positions(shape, occupied)
        valid = [(top, left, coords) for top, left, coords in positions if predicate(coords)]
        if not valid:
            continue
        top, left, coords = random.choice(valid)
        grid[coords[:, 0], coords[:, 1]] = int(color)
        occupied[coords[:, 0], coords[:, 1]] = True
        remaining.remove((shape, color))
        return True, {"shape": shape, "color": int(color), "top": top, "left": left, "coords": coords}
    return False, None


def _attempt_generate() -> np.ndarray | None:
    side = random.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((side, side), dtype=int)

    frame_size = random.choice(FRAME_SIZE_CHOICES)
    frame_corner = random.choice(["tl", "tr", "bl", "br"])
    frame_bbox = _frame_bbox(frame_corner, side, frame_size)

    template_size = random.randint(*TEMPLATE_SIZE_RANGE)
    template_shape = random.choice(SHAPES_BY_SIZE[template_size])
    object_count = random.randint(*OUTSIDE_OBJECT_COUNT_RANGE)
    distractor_count = object_count - 1
    distractor_shapes = _choose_distractor_shapes(template_shape, distractor_count)

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    random.shuffle(palette)
    template_color = palette[0]
    match_color = palette[1]
    distractor_colors = palette[2 : 2 + distractor_count]

    template_coords = _paint_frame_and_template(grid, frame_bbox, template_shape, template_color)
    occupied = np.zeros_like(grid, dtype=bool)
    fr0, fc0, fr1, fc1 = frame_bbox
    occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    match_candidates = []
    for _top, _left, coords in _candidate_positions(template_shape, occupied):
        ok, distance = _safe_match_position(coords, frame_bbox, grid.shape)
        if ok:
            match_candidates.append((coords, distance))
    if not match_candidates:
        return None

    coords_match, distance = random.choice(match_candidates)
    grid[coords_match[:, 0], coords_match[:, 1]] = match_color
    occupied[coords_match[:, 0], coords_match[:, 1]] = True
    ray_cells = _ray_cells(coords_match, distance, grid.shape)

    remaining = list(zip(distractor_shapes, distractor_colors))

    role_specs = [
        lambda coords: 0 < len({tuple(map(int, rc)) for rc in coords} & ray_cells) < len(coords),
        lambda coords: _distance_key(_bbox_from_coords(coords), frame_bbox) < distance,
        lambda coords: _distance_key(_bbox_from_coords(coords), frame_bbox) > distance,
    ]
    for predicate in role_specs:
        ok, _record = _place_role_object(grid, occupied, remaining, predicate)
        if not ok:
            return None

    random.shuffle(remaining)
    for shape, color in sorted(remaining, key=lambda item: -len(item[0])):
        positions = _candidate_positions(shape, occupied)
        if not positions:
            return None
        top, left, coords = random.choice(positions)
        grid[coords[:, 0], coords[:, 1]] = int(color)
        occupied[coords[:, 0], coords[:, 1]] = True

    scene = _parse_scene(grid)
    template_area = len(scene["template_shape"])
    template_dims = _shape_dims(scene["template_shape"])
    if not any(len(_shape_key(comp["coords"])) == template_area for comp in scene["distractors"]):
        return None
    if not any(_shape_dims(_shape_key(comp["coords"])) == template_dims for comp in scene["distractors"]):
        return None

    if not any(_distance_key(comp["bbox"], scene["frame"]["bbox"]) < distance for comp in scene["distractors"]):
        return None
    if not any(_distance_key(comp["bbox"], scene["frame"]["bbox"]) > distance for comp in scene["distractors"]):
        return None

    partial_overlap = False
    for comp in scene["distractors"]:
        cells = {tuple(map(int, rc)) for rc in comp["coords"]}
        overlap = len(cells & ray_cells)
        if 0 < overlap < len(cells):
            partial_overlap = True
            break
    if not partial_overlap:
        return None

    out = main(grid)
    protected = set()
    for comp in (scene["frame"], scene["template"], scene["match"]):
        protected |= {tuple(map(int, rc)) for rc in comp["coords"]}

    for cell in protected:
        if int(out[cell]) != int(grid[cell]):
            return None
    for comp in scene["distractors"]:
        for r, c in comp["coords"]:
            cell = (int(r), int(c))
            expected = RAY_COLOR if cell in ray_cells else BG
            if int(out[cell]) != expected:
                return None
    for cell in ray_cells:
        if cell not in protected and int(out[cell]) != RAY_COLOR:
            return None
    if np.array_equal(out, grid):
        return None

    colors = [int(scene["template"]["color"])] + [int(comp["color"]) for comp in scene["outside"]]
    if len(colors) != len(set(colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
