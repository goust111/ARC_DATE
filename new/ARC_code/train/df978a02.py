from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, is_contiguous
import random


def _central_background_cell(g: np.ndarray, bg: int, objects: list[dict]) -> tuple[int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    height, width = g.shape
    for r in range(height):
        for c in range(width):
            if int(g[r, c]) != bg:
                continue

            distances = []
            for obj in objects:
                coords = obj["coords"]
                d = np.abs(coords[:, 0] - r) + np.abs(coords[:, 1] - c)
                distances.append(int(d.min()))
            candidates.append((sum(distances), max(distances), r, c))

    assert candidates, "expected at least one background cell"
    candidates.sort()
    assert len(candidates) == 1 or candidates[0][:2] < candidates[1][:2], "expected a unique central gap"
    return int(candidates[0][2]), int(candidates[0][3])


def _inward_tip(obj: dict, center_r: int, center_c: int) -> tuple[int, int]:
    coords = obj["coords"]
    distances = np.abs(coords[:, 0] - center_r) + np.abs(coords[:, 1] - center_c)
    min_dist = int(distances.min())
    tips = coords[distances == min_dist]
    assert len(tips) == 1, "expected each losing object to have one inward tip"
    return int(tips[0, 0]), int(tips[0, 1])


def _outward_cap(obj: dict, center_r: int, center_c: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = obj["bbox"]
    obj_center_r = (r0 + r1) / 2
    obj_center_c = (c0 + c1) / 2
    row_offset = obj_center_r - center_r
    col_offset = obj_center_c - center_c

    if abs(row_offset) > abs(col_offset):
        assert row_offset != 0, "winner must sit away from the center"
        r = int(r0 - 1 if row_offset < 0 else r1 + 1)
        return [(r, center_c - 1), (r, center_c), (r, center_c + 1)]

    assert abs(col_offset) > abs(row_offset), "winner must have a cardinal outward direction"
    c = int(c0 - 1 if col_offset < 0 else c1 + 1)
    return [(center_r - 1, c), (center_r, c), (center_r + 1, c)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the largest center-pointing object grow outward while every other object loses its inward tip.

    Input structure
    - The grid has a single dominant background color and four 8-connected monochromatic foreground objects.
    - The objects surround a central background gap and point toward it from cardinal directions.
    - Exactly one object has the largest area.

    Rule
    - Find the central background gap by minimizing aggregate distance to the four objects.
    - Recolor the nearest-to-center pixel of each non-largest object to background.
    - Add a three-pixel cap in the largest object's color on the side away from the center, centered on the shared row or column.

    Key insight
    - The edit is a transfer of prominence around a common center: losers retract at their tips, and the unique largest object extends outward from the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert len(objects) == 4, "expected four colored objects around the center"

    center_r, center_c = _central_background_cell(g, bg, objects)

    areas = [int(obj["area"]) for obj in objects]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "expected a unique largest object"
    winner = objects[areas.index(max_area)]

    out = g.copy()
    for obj in objects:
        if obj is winner:
            continue
        r, c = _inward_tip(obj, center_r, center_c)
        out[r, c] = bg

    for r, c in _outward_cap(winner, center_r, center_c):
        assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "outward cap must stay in bounds"
        assert int(g[r, c]) == bg, "outward cap must be painted on background"
        out[r, c] = int(winner["color"])

    return out

# --------------- generator (c) ---------------

# Canvas and palette.
GRID_H, GRID_W = 16, 16
GEN_BG = 8
GEN_COLORS = [c for c in range(10) if c != GEN_BG]

# Center, direction, and gap sampling.
CENTER_MIN, CENTER_MAX = 5, 10
DIRECTIONS = ['above', 'below', 'left', 'right']
DIRECTION_STEPS = {'above': (-1, 0), 'below': (1, 0), 'left': (0, -1), 'right': (0, 1)}
VERTICAL_DIRECTIONS = ('above', 'below')
TIP_GAP_MIN, TIP_GAP_MAX = 1, 2

# Object area and growth bounds.
WINNER_AREA_MIN, WINNER_AREA_MAX = 8, 22
LOSER_AREA_MIN = 3
OBJECT_AREA_MIN, OBJECT_AREA_MAX = 3, 22
MAX_GROWTH_LAYERS, MAX_HALF_WIDTH_STEP = 10, 2

# Cap, spacing, and retry limits.
CAP_OFFSETS = (-1, 0, 1)
CAP_MARGIN, MIN_OBJECT_DISTANCE = 1, 2
GEN_RETRY_LIMIT = 300


def _tip_coord(cr: int, cc: int, direction: str, tip_gap: int) -> tuple[int, int]:
    step_r, step_c = DIRECTION_STEPS[direction]
    return cr + step_r * tip_gap, cc + step_c * tip_gap


def _perp_coord(direction: str, base_r: int, base_c: int, off: int) -> tuple[int, int]:
    if direction in VERTICAL_DIRECTIONS:
        return base_r, base_c + off
    return base_r + off, base_c


def _build_cardinal_object(cr: int, cc: int, direction: str, tip_gap: int,
                           target_area: int, rng: random.Random) -> list[tuple[int, int]] | None:
    tip = _tip_coord(cr, cc, direction, tip_gap)
    step_r, step_c = DIRECTION_STEPS[direction]

    coords = [tip]
    remaining = target_area - 1
    prev_hw = 0
    layer = 0

    while remaining > 0:
        layer += 1
        br = tip[0] + step_r * layer
        bc = tip[1] + step_c * layer

        min_hw = max(prev_hw, 1) if layer == 1 else max(prev_hw - 1, 0)
        max_hw = prev_hw + MAX_HALF_WIDTH_STEP
        feasible_hws = [hw for hw in range(min_hw, max_hw + 1) if 2 * hw + 1 <= remaining]
        if not feasible_hws:
            needed_hw = (remaining - 1) // 2
            if needed_hw >= min_hw and remaining == 2 * needed_hw + 1:
                feasible_hws = [needed_hw]
            else:
                break

        hw = rng.choice(feasible_hws)
        layer_pixels = [_perp_coord(direction, br, bc, off) for off in range(-hw, hw + 1)]
        coords.extend(layer_pixels)
        remaining -= len(layer_pixels)
        prev_hw = hw

        if layer > MAX_GROWTH_LAYERS:
            break

    if remaining > 0 and layer >= 1:
        br = tip[0] + step_r * layer
        bc = tip[1] + step_c * layer
        for side in [1, -1]:
            if remaining <= 0:
                break
            ext = prev_hw + 1
            p = _perp_coord(direction, br, bc, side * ext)
            coords.append(p)
            remaining -= 1

    return coords if len(coords) >= OBJECT_AREA_MIN else None


def _cap_coords(direction: str, cr: int, cc: int, obj_coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, c in obj_coords]
    cols = [c for r, c in obj_coords]
    if direction == 'above':
        return [(min(rows) - 1, cc + d) for d in CAP_OFFSETS]
    elif direction == 'below':
        return [(max(rows) + 1, cc + d) for d in CAP_OFFSETS]
    elif direction == 'left':
        return [(cr + d, min(cols) - 1) for d in CAP_OFFSETS]
    else:
        return [(cr + d, max(cols) + 1) for d in CAP_OFFSETS]


def _min_manhattan(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return min(abs(ra - rb) + abs(ca - cb) for ra, ca in a for rb, cb in b)


def _sample_parameters(rng: random.Random):
    cr = rng.randint(CENTER_MIN, CENTER_MAX)
    cc = rng.randint(CENTER_MIN, CENTER_MAX)

    directions = list(DIRECTIONS)
    rng.shuffle(directions)
    winner_dir = directions[0]
    loser_dirs = directions[1:]

    colors = rng.sample(GEN_COLORS, 4)
    winner_color, loser_colors = colors[0], colors[1:]

    winner_gap = rng.randint(TIP_GAP_MIN, TIP_GAP_MAX)
    loser_gaps = [rng.randint(TIP_GAP_MIN, TIP_GAP_MAX) for _ in range(3)]

    winner_area = rng.randint(WINNER_AREA_MIN, WINNER_AREA_MAX)
    loser_areas = [rng.randint(LOSER_AREA_MIN, winner_area - 1) for _ in range(3)]

    return (
        cr, cc, winner_dir, loser_dirs, winner_color, loser_colors,
        winner_gap, loser_gaps, winner_area, loser_areas,
    )


def _build_objects(cr: int, cc: int, winner_dir: str, loser_dirs: list[str], winner_gap: int,
                   loser_gaps: list[int], winner_area: int, loser_areas: list[int], rng: random.Random):
    winner_coords = _build_cardinal_object(cr, cc, winner_dir, winner_gap, winner_area, rng)
    if winner_coords is None:
        return None, None

    loser_coords_list = []
    for i in range(3):
        lc = _build_cardinal_object(cr, cc, loser_dirs[i], loser_gaps[i], loser_areas[i], rng)
        if lc is None:
            return None, None
        loser_coords_list.append(lc)
    return winner_coords, loser_coords_list


def _areas_are_valid(winner_coords: list[tuple[int, int]], loser_coords_list: list[list[tuple[int, int]]]) -> bool:
    actual_areas = [len(winner_coords)] + [len(lc) for lc in loser_coords_list]
    if any(a < OBJECT_AREA_MIN or a > OBJECT_AREA_MAX for a in actual_areas):
        return False
    return not any(len(lc) >= len(winner_coords) for lc in loser_coords_list)


def _points_fit_canvas(points: list[tuple[int, int]], cap: list[tuple[int, int]]) -> bool:
    if not all(0 <= r < GRID_H and 0 <= c < GRID_W for r, c in points):
        return False
    return all(CAP_MARGIN <= r < GRID_H - CAP_MARGIN and CAP_MARGIN <= c < GRID_W - CAP_MARGIN for r, c in cap)


def _groups_are_valid(obj_groups: list[list[tuple[int, int]]]) -> bool:
    for i in range(len(obj_groups)):
        if len(set(obj_groups[i])) != len(obj_groups[i]):
            return False
        temp = np.zeros((GRID_H, GRID_W), dtype=int)
        rows, cols = zip(*obj_groups[i])
        temp[rows, cols] = 1
        if not is_contiguous(temp, connectivity=8):
            return False
        for j in range(i + 1, 4):
            if _min_manhattan(obj_groups[i], obj_groups[j]) < MIN_OBJECT_DISTANCE:
                return False
    return True


def _paint(points: list[tuple[int, int]], grid: np.ndarray, color: int) -> None:
    rows, cols = zip(*points)
    grid[rows, cols] = color


def _render_pair(winner_coords: list[tuple[int, int]], cap: list[tuple[int, int]],
                 loser_coords_list: list[list[tuple[int, int]]], winner_color: int,
                 loser_colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((GRID_H, GRID_W), GEN_BG, dtype=int)
    _paint(winner_coords, inp, winner_color)
    for i, lc in enumerate(loser_coords_list):
        _paint(lc, inp, loser_colors[i])

    output = np.full((GRID_H, GRID_W), GEN_BG, dtype=int)
    _paint(winner_coords + cap, output, winner_color)
    for i, lc in enumerate(loser_coords_list):
        _paint(lc[1:], output, loser_colors[i])
    return inp, output


def _oracle_accepts(inp: np.ndarray, output: np.ndarray) -> bool:
    try:
        return np.array_equal(main(inp), output)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    (
        cr, cc, winner_dir, loser_dirs, winner_color, loser_colors,
        winner_gap, loser_gaps, winner_area, loser_areas,
    ) = _sample_parameters(rng)

    winner_coords, loser_coords_list = _build_objects(
        cr, cc, winner_dir, loser_dirs, winner_gap, loser_gaps, winner_area, loser_areas, rng
    )
    if winner_coords is None or loser_coords_list is None:
        return None
    if not _areas_are_valid(winner_coords, loser_coords_list):
        return None

    cap = _cap_coords(winner_dir, cr, cc, winner_coords)
    obj_groups = [winner_coords + cap] + loser_coords_list
    all_pts = []
    for grp in obj_groups:
        all_pts += grp
    if not _points_fit_canvas(all_pts, cap):
        return None
    if not _groups_are_valid(obj_groups):
        return None

    inp, output = _render_pair(winner_coords, cap, loser_coords_list, winner_color, loser_colors)
    if not _oracle_accepts(inp, output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRY_LIMIT} attempts")
