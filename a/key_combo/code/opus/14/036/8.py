from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
INPUT_COLORS = (1, 2, 3, 4)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Base templates use a top-side tip at (0, 0) and grow outward/upward.
BASE_TEMPLATES = {
    4: [(0, 0), (-1, 0), (-2, 0), (-1, 1)],
    5: [(0, 0), (-1, 0), (-2, 0), (-1, 1), (-1, -1)],
    6: [(0, 0), (-1, 0), (-2, 0), (-1, 1), (-1, -1), (-2, 1)],
    7: [(0, 0), (-1, 0), (-2, 0), (-1, 1), (-1, -1), (-2, 1), (-2, -1)],
    8: [(0, 0), (-1, 0), (-2, 0), (-1, 1), (-1, -1), (-2, 1), (-2, -1), (-3, 0)],
    9: [(0, 0), (-1, 0), (-2, 0), (-1, 1), (-1, -1), (-2, 1), (-2, -1), (-3, 0), (-3, 1)],
    10: [
        (0, 0),
        (-1, 0),
        (-2, 0),
        (-1, 1),
        (-1, -1),
        (-2, 1),
        (-2, -1),
        (-3, 0),
        (-3, 1),
        (-3, -1),
    ],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": _bbox_from_coords(coord_arr),
                    "area": int(coord_arr.shape[0]),
                }
            )
    return objects


def _bbox_center(obj: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = obj["bbox"]
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _classify_sides(objects: list[dict]) -> dict[str, dict]:
    assert len(objects) == 4, "expected exactly four objects"
    by_row = sorted(objects, key=lambda obj: (_bbox_center(obj)[0], _bbox_center(obj)[1]))
    top = by_row[0]
    bottom = by_row[-1]
    middle = [obj for obj in objects if obj is not top and obj is not bottom]
    assert len(middle) == 2, "side classification failed"
    left, right = sorted(middle, key=lambda obj: (_bbox_center(obj)[1], _bbox_center(obj)[0]))
    side_map = {"top": top, "bottom": bottom, "left": left, "right": right}

    row_centers = {_bbox_center(obj)[0] for obj in objects}
    col_centers = {_bbox_center(obj)[1] for obj in objects}
    assert len(row_centers) >= 3 and len(col_centers) >= 3, "objects must occupy four cardinal roles"
    return side_map


def _center_gap_bbox(side_map: dict[str, dict], grid: np.ndarray) -> tuple[int, int, int, int]:
    top = side_map["top"]["bbox"]
    bottom = side_map["bottom"]["bbox"]
    left = side_map["left"]["bbox"]
    right = side_map["right"]["bbox"]

    r0 = top[2] + 1
    r1 = bottom[0] - 1
    c0 = left[3] + 1
    c1 = right[1] - 1
    assert r0 <= r1 and c0 <= c1, "center gap must be non-empty"
    gap = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(gap == BLACK), "center gap must be background"
    return r0, c0, r1, c1


def _dist_to_gap_cell(r: int, c: int, gap_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = gap_bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def _tip_pixel(obj: dict, gap_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    distances = []
    for r, c in obj["coords"]:
        rr, cc = int(r), int(c)
        distances.append((_dist_to_gap_cell(rr, cc, gap_bbox), rr, cc))
    distances.sort()
    assert len(distances) == 1 or distances[0][0] < distances[1][0], "tip pixel must be unique"
    return distances[0][1], distances[0][2]


def _cap_cells(obj: dict, side: str) -> list[tuple[int, int]]:
    coords = {(int(r), int(c)) for r, c in obj["coords"]}
    r0, c0, r1, c1 = obj["bbox"]

    if side == "top":
        edge_cols = sorted(c for r, c in coords if r == r0)
        mid_c = edge_cols[len(edge_cols) // 2]
        rr = r0 - 1
        return [(rr, mid_c - 1), (rr, mid_c), (rr, mid_c + 1)]
    if side == "bottom":
        edge_cols = sorted(c for r, c in coords if r == r1)
        mid_c = edge_cols[len(edge_cols) // 2]
        rr = r1 + 1
        return [(rr, mid_c - 1), (rr, mid_c), (rr, mid_c + 1)]
    if side == "left":
        edge_rows = sorted(r for r, c in coords if c == c0)
        mid_r = edge_rows[len(edge_rows) // 2]
        cc = c0 - 1
        return [(mid_r - 1, cc), (mid_r, cc), (mid_r + 1, cc)]

    edge_rows = sorted(r for r, c in coords if c == c1)
    mid_r = edge_rows[len(edge_rows) // 2]
    cc = c1 + 1
    return [(mid_r - 1, cc), (mid_r, cc), (mid_r + 1, cc)]


def _ray_cells(start: tuple[int, int], side: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = SIDE_TO_DIR[side]
    r, c = start
    cells = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _independent_transform(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert Counter(map(int, g.ravel())).most_common(1)[0][0] == BLACK

    objects = _find_components(g, bg=BLACK)
    assert len(objects) == 4, "expected four monochromatic objects"
    assert {obj["color"] for obj in objects} == set(INPUT_COLORS), "expected input colors 1..4"

    side_map = _classify_sides(objects)
    gap_bbox = _center_gap_bbox(side_map, g)

    areas = {side: int(obj["area"]) for side, obj in side_map.items()}
    largest = max(areas.values())
    assert list(areas.values()).count(largest) == 1, "largest area must be unique"
    winner_side = next(side for side, area in areas.items() if area == largest)
    winner = side_map[winner_side]

    out = g.copy()
    tip_by_side: dict[str, tuple[int, int]] = {}
    ray_by_side: dict[str, list[tuple[int, int]]] = {}

    for side, obj in side_map.items():
        tip = _tip_pixel(obj, gap_bbox)
        tip_by_side[side] = tip
        if side != winner_side:
            out[tip] = BLACK

    for rr, cc in _cap_cells(winner, winner_side):
        assert _in_bounds(g.shape, rr, cc), "winner cap must stay in bounds"
        assert int(g[rr, cc]) == BLACK, "winner cap must land on background"
        out[rr, cc] = int(winner["color"])

    for side in ("top", "bottom", "left", "right"):
        if side == winner_side:
            continue
        ray = _ray_cells(tip_by_side[side], side, g.shape)
        ray_by_side[side] = ray
        for rr, cc in ray:
            out[rr, cc] = GREY

    intersections: list[tuple[int, int]] = []
    vertical_sides = [side for side in ("top", "bottom") if side in ray_by_side]
    horizontal_sides = [side for side in ("left", "right") if side in ray_by_side]
    for v_side in vertical_sides:
        v_set = set(ray_by_side[v_side])
        for h_side in horizontal_sides:
            h_set = set(ray_by_side[h_side])
            intersections.extend(sorted(v_set & h_set))

    intersections = sorted(set(intersections))
    for rr, cc in intersections:
        out[rr, cc] = ORANGE

    for rr, cc in intersections:
        rr += 1
        cc -= 1
        while _in_bounds(g.shape, rr, cc) and int(out[rr, cc]) == BLACK:
            out[rr, cc] = ORANGE
            rr += 1
            cc -= 1

    meta = {
        "objects": objects,
        "side_map": side_map,
        "gap_bbox": gap_bbox,
        "winner_side": winner_side,
        "winner_color": int(winner["color"]),
        "tip_by_side": tip_by_side,
        "ray_by_side": ray_by_side,
        "intersections": intersections,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink the three non-largest center-facing objects into grey outward rays and let the unique largest object grow an outward cap.

    Input structure:
    - The background is black, and there are exactly four 4-connected monochromatic foreground objects in four distinct colors.
    - The four objects lie strictly above, below, left, and right of a rectangular central black gap.
    - Each object has a unique pixel closest to that central gap, and the four object areas are all different.

    Rule:
    - Identify the top, bottom, left, and right objects and recover the central gap from the empty rectangle between them.
    - Find the unique largest-area object; add a 3-pixel outward cap one step beyond its outer edge, centered on that edge.
    - For each other object, remove its unique nearest-to-gap tip from the object and repaint a grey ray from that released tip outward to the grid boundary.
    - Wherever a grey vertical ray and a grey horizontal ray cross, recolor the crossing orange and continue an orange down-left diagonal until a non-black cell or the grid edge stops it.

    Key insight:
    - Area comparison chooses the only grower, while each shrinking object's released inward tip becomes the source of a new outward ray whose crossings create the final orange structure.
    """
    out, _meta = _independent_transform(np.asarray(grid, dtype=int))
    return out


def _transform_template(base: list[tuple[int, int]], side: str, mirror: bool) -> list[tuple[int, int]]:
    rels = [(dr, -dc) if mirror else (dr, dc) for dr, dc in base]
    out = []
    for dr, dc in rels:
        if side == "top":
            rr, cc = dr, dc
        elif side == "bottom":
            rr, cc = -dr, -dc
        elif side == "left":
            rr, cc = -dc, dr
        else:
            rr, cc = dc, -dr
        out.append((rr, cc))
    return out


def _render_object(side: str, tip: tuple[int, int], area: int, mirror: bool) -> list[tuple[int, int]]:
    base = BASE_TEMPLATES[area]
    rels = _transform_template(base, side, mirror)
    tr, tc = tip
    return [(tr + dr, tc + dc) for dr, dc in rels]


def _choose_gap(h: int, w: int) -> tuple[int, int, int, int]:
    gap_h = random.randint(2, 4)
    gap_w = random.randint(2, 4)
    r0 = random.randint(5, h - gap_h - 6)
    c0 = random.randint(5, w - gap_w - 6)
    return r0, c0, r0 + gap_h - 1, c0 + gap_w - 1


def _base_tips(gap_bbox: tuple[int, int, int, int], collision_pair: tuple[str, str] | None) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = gap_bbox
    tips = {
        "top": (r0 - random.randint(1, 2), random.randint(c0, c1)),
        "bottom": (r1 + random.randint(1, 2), random.randint(c0, c1)),
        "left": (random.randint(r0, r1), c0 - random.randint(1, 2)),
        "right": (random.randint(r0, r1), c1 + random.randint(1, 2)),
    }

    if collision_pair is None:
        return tips

    pair = tuple(sorted(collision_pair))
    if pair == ("left", "top"):
        left_col = c0 - random.randint(2, 3)
        top_row = r0 - random.randint(2, 3)
        tips["left"] = (top_row - 1, left_col)
        tips["top"] = (top_row, left_col - 1)
    elif pair == ("right", "top"):
        right_col = c1 + random.randint(2, 3)
        top_row = r0 - random.randint(2, 3)
        tips["right"] = (top_row - 1, right_col)
        tips["top"] = (top_row, right_col + 1)
    elif pair == ("bottom", "left"):
        left_col = c0 - random.randint(2, 3)
        bottom_row = r1 + random.randint(2, 3)
        tips["left"] = (bottom_row + 1, left_col)
        tips["bottom"] = (bottom_row, left_col - 1)
    elif pair == ("bottom", "right"):
        right_col = c1 + random.randint(2, 3)
        bottom_row = r1 + random.randint(2, 3)
        tips["right"] = (bottom_row + 1, right_col)
        tips["bottom"] = (bottom_row, right_col + 1)
    else:
        raise AssertionError("unsupported collision pair")
    return tips


def _pair_candidates(winner_side: str) -> list[tuple[str, str]]:
    all_pairs = [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    return [pair for pair in all_pairs if winner_side not in pair]


def _place_objects(
    h: int,
    w: int,
    gap_bbox: tuple[int, int, int, int],
    winner_side: str,
    desired_collision: bool,
) -> np.ndarray | None:
    area_values = random.sample(list(BASE_TEMPLATES), 4)
    area_values.sort()
    winner_area = area_values[-1]
    loser_areas = area_values[:-1]
    random.shuffle(loser_areas)

    side_to_area = {winner_side: winner_area}
    loser_sides = [side for side in ("top", "bottom", "left", "right") if side != winner_side]
    for side, area in zip(loser_sides, loser_areas):
        side_to_area[side] = area

    collision_pair = random.choice(_pair_candidates(winner_side)) if desired_collision else None
    tips = _base_tips(gap_bbox, collision_pair)
    side_to_color = dict(zip(random.sample(("top", "bottom", "left", "right"), 4), INPUT_COLORS))

    objects: dict[str, list[tuple[int, int]]] = {}
    for side in ("top", "bottom", "left", "right"):
        coords = _render_object(side, tips[side], side_to_area[side], mirror=bool(random.getrandbits(1)))
        objects[side] = coords

    g = np.zeros((h, w), dtype=int)
    r0, c0, r1, c1 = gap_bbox
    occupied: dict[tuple[int, int], str] = {}
    for side, coords in objects.items():
        for rr, cc in coords:
            if not _in_bounds(g.shape, rr, cc):
                return None
            if side == "top" and rr >= r0:
                return None
            if side == "bottom" and rr <= r1:
                return None
            if side == "left" and cc >= c0:
                return None
            if side == "right" and cc <= c1:
                return None
            if (rr, cc) in occupied:
                return None
            occupied[(rr, cc)] = side

    for side, coords in objects.items():
        color = side_to_color[side]
        for rr, cc in coords:
            g[rr, cc] = color

    if np.any(g[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
        return None
    return g


def _validate_example(inp: np.ndarray, desired_collision: bool) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2 or max(inp.shape) > 30:
        return False

    try:
        out, meta = _independent_transform(inp)
    except Exception:
        return False

    if np.array_equal(inp, out):
        return False

    side_map = meta["side_map"]
    winner_side = meta["winner_side"]
    winner_color = meta["winner_color"]
    intersections = meta["intersections"]

    if desired_collision != bool(intersections):
        return False
    if desired_collision and np.count_nonzero(out == ORANGE) < 2:
        return False

    winner_in = int(np.count_nonzero(inp == winner_color))
    winner_out = int(np.count_nonzero(out == winner_color))
    if winner_out != winner_in + 3:
        return False

    for side, obj in side_map.items():
        color = int(obj["color"])
        if side == winner_side:
            continue
        tip = meta["tip_by_side"][side]
        if int(out[tip]) != GREY:
            return False
        remaining = int(np.count_nonzero(out == color))
        if remaining <= 0:
            return False

    cap_cells = _cap_cells(side_map[winner_side], winner_side)
    for rr, cc in cap_cells:
        if not _in_bounds(inp.shape, rr, cc) or int(inp[rr, cc]) != BLACK or int(out[rr, cc]) != winner_color:
            return False

    objects = _find_components(inp, bg=BLACK)
    if len(objects) != 4:
        return False
    areas = sorted(obj["area"] for obj in objects)
    if len(set(areas)) != 4:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(16, 20)
        w = random.randint(16, 20)
        gap_bbox = _choose_gap(h, w)
        winner_side = random.choice(("top", "bottom", "left", "right"))
        desired_collision = random.random() < 0.5
        inp = _place_objects(h, w, gap_bbox, winner_side, desired_collision)
        if inp is None:
            continue
        if _validate_example(inp, desired_collision):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 8")
