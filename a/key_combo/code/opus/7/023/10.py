from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
MARKER_COLORS = (RED, GREEN)
SIDES = ("top", "bottom", "left", "right")
DIRS = ("up", "down", "left", "right")

GEN_ATTEMPTS = 400
GRID_MIN = 20
GRID_MAX = 24
ZONE_MARGIN = 1
ZONE_GAP = 3


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_center2(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r0 + r1, c0 + c1


def _bbox_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int]:
    ra0, ca0, ra1, ca1 = box_a
    rb0, cb0, rb1, cb1 = box_b
    row_gap = max(0, rb0 - ra1 - 1, ra0 - rb1 - 1)
    col_gap = max(0, cb0 - ca1 - 1, ca0 - cb1 - 1)
    return int(row_gap), int(col_gap)


def _detect_marker(
    grid: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[str, int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    candidates: list[tuple[str, int, list[tuple[int, int]]]] = []

    if r0 > 0:
        vals = grid[r0 - 1, c0 : c1 + 1]
        if vals.size > 0 and int(vals[0]) in MARKER_COLORS and np.all(vals == vals[0]):
            candidates.append(("top", int(vals[0]), [(r0 - 1, c) for c in range(c0, c1 + 1)]))
    if r1 + 1 < grid.shape[0]:
        vals = grid[r1 + 1, c0 : c1 + 1]
        if vals.size > 0 and int(vals[0]) in MARKER_COLORS and np.all(vals == vals[0]):
            candidates.append(("bottom", int(vals[0]), [(r1 + 1, c) for c in range(c0, c1 + 1)]))
    if c0 > 0:
        vals = grid[r0 : r1 + 1, c0 - 1]
        if vals.size > 0 and int(vals[0]) in MARKER_COLORS and np.all(vals == vals[0]):
            candidates.append(("left", int(vals[0]), [(r, c0 - 1) for r in range(r0, r1 + 1)]))
    if c1 + 1 < grid.shape[1]:
        vals = grid[r0 : r1 + 1, c1 + 1]
        if vals.size > 0 and int(vals[0]) in MARKER_COLORS and np.all(vals == vals[0]):
            candidates.append(("right", int(vals[0]), [(r, c1 + 1) for r in range(r0, r1 + 1)]))

    assert len(candidates) == 1, f"expected exactly one side marker for bbox {box}, got {candidates}"
    return candidates[0]


def _parse_objects(grid: np.ndarray) -> list[dict]:
    objs: list[dict] = []
    for coords in _components(grid == BLUE):
        box = _bbox(coords)
        side, marker_color, marker_coords = _detect_marker(grid, box)
        objs.append(
            {
                "coords": coords,
                "bbox": box,
                "area": int(coords.shape[0]),
                "side": side,
                "marker_color": int(marker_color),
                "marker_coords": marker_coords,
            }
        )
    return objs


def _dominant_direction(src_box: tuple[int, int, int, int], dst_box: tuple[int, int, int, int]) -> str:
    src_r2, src_c2 = _bbox_center2(src_box)
    dst_r2, dst_c2 = _bbox_center2(dst_box)
    dr2 = dst_r2 - src_r2
    dc2 = dst_c2 - src_c2
    assert dr2 != 0 or dc2 != 0, "source and target objects must be distinct"
    assert abs(dr2) != abs(dc2), "main-axis direction would be ambiguous"
    if abs(dr2) > abs(dc2):
        return "down" if dr2 > 0 else "up"
    return "right" if dc2 > 0 else "left"


def _paint_extension(out: np.ndarray, src_box: tuple[int, int, int, int], dst_box: tuple[int, int, int, int]) -> int:
    direction = _dominant_direction(src_box, dst_box)
    sr0, sc0, sr1, sc1 = src_box
    tr0, tc0, tr1, tc1 = dst_box

    if direction == "down":
        view = out[sr0 : tr0 + 1, sc0 : sc1 + 1]
    elif direction == "up":
        view = out[tr1 : sr1 + 1, sc0 : sc1 + 1]
    elif direction == "right":
        view = out[sr0 : sr1 + 1, sc0 : tc0 + 1]
    else:
        view = out[sr0 : sr1 + 1, tc1 : sc1 + 1]

    added = int(np.count_nonzero(view == BG))
    view[view == BG] = BLUE
    return added


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the smallest side-marked object in each color group, then extend the largest object toward it.

    Input structure:
    - Black background contains 4-6 blue 4-connected objects, each separated from the others.
    - Every blue object's bounding box has exactly one adjacent one-pixel marker strip on one side.
    - Marker strips are red or green; the strip color groups the blue objects.

    Rule:
    - For each blue object, read the color of its unique adjacent side strip.
    - Within each marker-color group, rank objects by blue-pixel area.
    - Erase the unique smallest object together with its strip.
    - From the unique largest object's bounding box, extend a blue rectangle toward the erased object along the dominant axis between their bbox centers; the rectangle keeps the largest bbox's orthogonal span and reaches the erased bbox's near edge, but only background cells are painted.

    Key insight:
    - The side strip is only a grouping label: group membership picks which object dies and which one expands, while the erased object's position determines the expansion direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _parse_objects(g)
    assert 4 <= len(objects) <= 6, "expected 4-6 blue objects"

    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        groups[obj["marker_color"]].append(obj)

    assert set(groups) == {RED, GREEN}, f"expected red and green groups, got {sorted(groups)}"

    out = g.copy()
    for marker_color, members in groups.items():
        assert len(members) >= 2, f"group {marker_color} must have at least two objects"
        areas = [int(obj["area"]) for obj in members]
        assert len(set(areas)) == len(areas), f"group {marker_color} areas must be unique"

        ranked = sorted(members, key=lambda obj: int(obj["area"]), reverse=True)
        largest = ranked[0]
        smallest = ranked[-1]

        for r, c in largest["marker_coords"]:
            assert out[r, c] == marker_color
        for r, c in smallest["marker_coords"]:
            out[r, c] = BG
        out[smallest["coords"][:, 0], smallest["coords"][:, 1]] = BG
        _paint_extension(out, largest["bbox"], smallest["bbox"])

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _grow_polyomino(area: int) -> tuple[list[tuple[int, int]], int, int]:
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        rel = _normalize_cells(cells)
        rows = [r for r, _ in rel]
        cols = [c for _, c in rel]
        h = max(rows) + 1
        w = max(cols) + 1
        density = area / float(h * w)
        if h < 2 or w < 2 or h > 5 or w > 5 or density < 0.45:
            continue
        if area != h * w and random.random() < 0.75:
            return rel, h, w
        if random.random() < 0.2:
            return rel, h, w
    raise RuntimeError(f"failed to build polyomino of area {area}")


def _footprint_bbox(obj: dict, r: int, c: int) -> tuple[int, int, int, int]:
    h = int(obj["h"])
    w = int(obj["w"])
    side = str(obj["side"])
    if side == "top":
        return r - 1, c, r + h - 1, c + w - 1
    if side == "bottom":
        return r, c, r + h, c + w - 1
    if side == "left":
        return r, c - 1, r + h - 1, c + w - 1
    return r, c, r + h - 1, c + w


def _expand_bbox(
    box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _mark_occupied(occupied: np.ndarray, obj: dict, r: int, c: int) -> None:
    occ_box = _expand_bbox(_footprint_bbox(obj, r, c), 1, occupied.shape)
    r0, c0, r1, c1 = occ_box
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _bbox_from_pos(obj: dict, r: int, c: int) -> tuple[int, int, int, int]:
    return r, c, r + int(obj["h"]) - 1, c + int(obj["w"]) - 1


def _bbox_overlaps_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _candidate_positions(
    obj: dict,
    zone: tuple[int, int, int, int],
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    oh, ow = int(obj["h"]), int(obj["w"])
    zr0, zc0, zr1, zc1 = zone
    positions: list[tuple[int, int]] = []
    for r in range(h - oh + 1):
        for c in range(w - ow + 1):
            fbox = _footprint_bbox(obj, r, c)
            if not (zr0 <= fbox[0] and fbox[2] <= zr1 and zc0 <= fbox[1] and fbox[3] <= zc1):
                continue
            r0, c0, r1, c1 = _expand_bbox(fbox, 1, occupied.shape)
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            positions.append((r, c))
    random.shuffle(positions)
    return positions


def _direction_relation(
    src_box: tuple[int, int, int, int], dst_box: tuple[int, int, int, int], direction: str
) -> bool:
    src_r2, src_c2 = _bbox_center2(src_box)
    dst_r2, dst_c2 = _bbox_center2(dst_box)
    dr2 = dst_r2 - src_r2
    dc2 = dst_c2 - src_c2
    if direction == "down":
        return dst_box[0] >= src_box[2] + 3 and dr2 >= abs(dc2) + 2
    if direction == "up":
        return src_box[0] >= dst_box[2] + 3 and -dr2 >= abs(dc2) + 2
    if direction == "right":
        return dst_box[1] >= src_box[3] + 3 and dc2 >= abs(dr2) + 2
    return src_box[1] >= dst_box[3] + 3 and -dc2 >= abs(dr2) + 2


def _corridor_relation(
    mid_box: tuple[int, int, int, int],
    src_box: tuple[int, int, int, int],
    dst_box: tuple[int, int, int, int],
    direction: str,
) -> bool:
    mid_r2, mid_c2 = _bbox_center2(mid_box)
    src_r2, src_c2 = _bbox_center2(src_box)
    dst_r2, dst_c2 = _bbox_center2(dst_box)
    if direction in ("down", "up"):
        return (
            min(src_r2, dst_r2) < mid_r2 < max(src_r2, dst_r2)
            and _bbox_overlaps_1d(mid_box[1], mid_box[3], src_box[1], src_box[3])
        )
    return (
        min(src_c2, dst_c2) < mid_c2 < max(src_c2, dst_c2)
        and _bbox_overlaps_1d(mid_box[0], mid_box[2], src_box[0], src_box[2])
    )


def _choose_pair(
    largest: dict,
    smallest: dict,
    zone: tuple[int, int, int, int],
    occupied: np.ndarray,
    direction: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for src_pos in _candidate_positions(largest, zone, occupied):
        temp = occupied.copy()
        _mark_occupied(temp, largest, *src_pos)
        src_box = _bbox_from_pos(largest, *src_pos)
        dst_positions = _candidate_positions(smallest, zone, temp)
        good = [pos for pos in dst_positions if _direction_relation(src_box, _bbox_from_pos(smallest, *pos), direction)]
        if good:
            return src_pos, random.choice(good)
    return None


def _choose_middle(
    obj: dict,
    zone: tuple[int, int, int, int],
    occupied: np.ndarray,
    src_box: tuple[int, int, int, int],
    dst_box: tuple[int, int, int, int],
    direction: str,
    require_corridor: bool,
) -> tuple[int, int] | None:
    positions = _candidate_positions(obj, zone, occupied)
    if not require_corridor:
        return positions[0] if positions else None
    good = [pos for pos in positions if _corridor_relation(_bbox_from_pos(obj, *pos), src_box, dst_box, direction)]
    return random.choice(good) if good else None


def _render_marker(grid: np.ndarray, obj: dict, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    marker_color = int(obj["marker_color"])
    if obj["side"] == "top":
        grid[r0 - 1, c0 : c1 + 1] = marker_color
    elif obj["side"] == "bottom":
        grid[r1 + 1, c0 : c1 + 1] = marker_color
    elif obj["side"] == "left":
        grid[r0 : r1 + 1, c0 - 1] = marker_color
    else:
        grid[r0 : r1 + 1, c1 + 1] = marker_color


def _render_scene(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in placed:
        r0, c0, r1, c1 = obj["bbox"]
        for rr, cc in obj["shape"]:
            grid[r0 + rr, c0 + cc] = BLUE
        _render_marker(grid, obj, obj["bbox"])
    return grid


def _expected_from_metadata(grid: np.ndarray, placed: list[dict]) -> tuple[np.ndarray, dict[int, int]]:
    out = grid.copy()
    per_group_added: dict[int, int] = {}
    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in placed:
        groups[int(obj["marker_color"])].append(obj)

    for marker_color, members in groups.items():
        ranked = sorted(members, key=lambda obj: int(obj["area"]), reverse=True)
        largest = ranked[0]
        smallest = ranked[-1]
        sr0, sc0, sr1, sc1 = smallest["bbox"]
        out[smallest["coords_abs"][:, 0], smallest["coords_abs"][:, 1]] = BG
        if smallest["side"] == "top":
            out[sr0 - 1, sc0 : sc1 + 1] = BG
        elif smallest["side"] == "bottom":
            out[sr1 + 1, sc0 : sc1 + 1] = BG
        elif smallest["side"] == "left":
            out[sr0 : sr1 + 1, sc0 - 1] = BG
        else:
            out[sr0 : sr1 + 1, sc1 + 1] = BG
        per_group_added[marker_color] = _paint_extension(out, largest["bbox"], smallest["bbox"])
    return out, per_group_added


def _assign_sides(groups: dict[int, list[dict]]) -> bool:
    all_objects = [obj for members in groups.values() for obj in members]
    for _ in range(120):
        for obj in all_objects:
            obj["side"] = random.choice(SIDES)
        sides_by_color = {color: {obj["side"] for obj in members} for color, members in groups.items()}
        if len({obj["side"] for obj in all_objects}) < 3:
            continue
        if not (sides_by_color[RED] & sides_by_color[GREEN]):
            continue
        if not any(len(sides) >= 2 for sides in sides_by_color.values()):
            continue
        return True
    return False


def _build_group(area_list: list[int], marker_color: int) -> list[dict]:
    members = []
    for area in sorted(area_list, reverse=True):
        shape, h, w = _grow_polyomino(area)
        members.append(
            {
                "area": int(area),
                "shape": shape,
                "h": int(h),
                "w": int(w),
                "marker_color": int(marker_color),
            }
        )
    return members


def _make_zones(h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    if random.random() < 0.5:
        cut = random.randint(w // 2 - 2, w // 2 + 2)
        left = (ZONE_MARGIN, ZONE_MARGIN, h - 1 - ZONE_MARGIN, cut - 2)
        right = (ZONE_MARGIN, cut + 2, h - 1 - ZONE_MARGIN, w - 1 - ZONE_MARGIN)
        if left[3] - left[1] + 1 < 8 or right[3] - right[1] + 1 < 8:
            return None
        return left, right
    cut = random.randint(h // 2 - 2, h // 2 + 2)
    top = (ZONE_MARGIN, ZONE_MARGIN, cut - 2, w - 1 - ZONE_MARGIN)
    bottom = (cut + 2, ZONE_MARGIN, h - 1 - ZONE_MARGIN, w - 1 - ZONE_MARGIN)
    if top[2] - top[0] + 1 < 8 or bottom[2] - bottom[0] + 1 < 8:
        return None
    return top, bottom


def _place_group(
    members: list[dict],
    zone: tuple[int, int, int, int],
    occupied: np.ndarray,
    direction: str,
    require_corridor: bool,
) -> list[dict] | None:
    largest = members[0]
    smallest = members[-1]
    pair = _choose_pair(largest, smallest, zone, occupied, direction)
    if pair is None:
        return None
    src_pos, dst_pos = pair

    placed: list[dict] = []

    largest_box = _bbox_from_pos(largest, *src_pos)
    largest_abs = np.array([(largest_box[0] + r, largest_box[1] + c) for r, c in largest["shape"]], dtype=int)
    placed_largest = {**largest, "bbox": largest_box, "coords_abs": largest_abs}
    placed.append(placed_largest)
    _mark_occupied(occupied, largest, *src_pos)

    smallest_box = _bbox_from_pos(smallest, *dst_pos)
    smallest_abs = np.array([(smallest_box[0] + r, smallest_box[1] + c) for r, c in smallest["shape"]], dtype=int)
    placed_smallest = {**smallest, "bbox": smallest_box, "coords_abs": smallest_abs}
    _mark_occupied(occupied, smallest, *dst_pos)

    for idx, mid in enumerate(members[1:-1]):
        mid_pos = _choose_middle(
            mid,
            zone,
            occupied,
            largest_box,
            smallest_box,
            direction,
            require_corridor and idx == 0,
        )
        if mid_pos is None:
            return None
        mid_box = _bbox_from_pos(mid, *mid_pos)
        mid_abs = np.array([(mid_box[0] + r, mid_box[1] + c) for r, c in mid["shape"]], dtype=int)
        placed.append({**mid, "bbox": mid_box, "coords_abs": mid_abs})
        _mark_occupied(occupied, mid, *mid_pos)

    placed.append(placed_smallest)
    return placed


def _validate_scene(
    grid: np.ndarray,
    placed: list[dict],
    directions: dict[int, str],
    corridor_color: int | None,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    parsed = _parse_objects(grid)
    if len(parsed) != len(placed):
        return False

    parsed_groups: dict[int, list[dict]] = defaultdict(list)
    for obj in parsed:
        parsed_groups[int(obj["marker_color"])].append(obj)
    if set(parsed_groups) != {RED, GREEN}:
        return False

    for members in parsed_groups.values():
        if len(members) < 2:
            return False
        if len({int(obj["area"]) for obj in members}) != len(members):
            return False

    red_areas = sorted([int(obj["area"]) for obj in parsed_groups[RED]])
    green_areas = sorted([int(obj["area"]) for obj in parsed_groups[GREEN]])
    if not (max(red_areas) < min(green_areas) or max(green_areas) < min(red_areas)):
        return False

    side_sets = {color: {obj["side"] for obj in parsed_groups[color]} for color in (RED, GREEN)}
    if len(side_sets[RED] | side_sets[GREEN]) < 3:
        return False
    if not (side_sets[RED] & side_sets[GREEN]):
        return False

    bboxes = [obj["bbox"] for obj in parsed]
    for i, box_a in enumerate(bboxes):
        for box_b in bboxes[i + 1 :]:
            row_gap, col_gap = _bbox_gap(box_a, box_b)
            if max(row_gap, col_gap) < 2:
                return False

    expected, per_group_added = _expected_from_metadata(grid, placed)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if any(v <= 0 for v in per_group_added.values()):
        return False

    for color, members in parsed_groups.items():
        ranked = sorted(members, key=lambda obj: int(obj["area"]), reverse=True)
        if _dominant_direction(ranked[0]["bbox"], ranked[-1]["bbox"]) != directions[color]:
            return False

    if corridor_color is not None:
        ranked = sorted(
            [obj for obj in placed if int(obj["marker_color"]) == corridor_color],
            key=lambda obj: int(obj["area"]),
            reverse=True,
        )
        largest = ranked[0]
        smallest = ranked[-1]
        temp = grid.copy()
        temp[smallest["coords_abs"][:, 0], smallest["coords_abs"][:, 1]] = BG
        sr0, sc0, sr1, sc1 = smallest["bbox"]
        if smallest["side"] == "top":
            temp[sr0 - 1, sc0 : sc1 + 1] = BG
        elif smallest["side"] == "bottom":
            temp[sr1 + 1, sc0 : sc1 + 1] = BG
        elif smallest["side"] == "left":
            temp[sr0 : sr1 + 1, sc0 - 1] = BG
        else:
            temp[sr0 : sr1 + 1, sc1 + 1] = BG

        direction = directions[corridor_color]
        lr0, lc0, lr1, lc1 = largest["bbox"]
        if direction == "down":
            rect = temp[lr0 : sr0 + 1, lc0 : lc1 + 1]
        elif direction == "up":
            rect = temp[sr1 : lr1 + 1, lc0 : lc1 + 1]
        elif direction == "right":
            rect = temp[lr0 : lr1 + 1, lc0 : sc0 + 1]
        else:
            rect = temp[lr0 : lr1 + 1, sc1 : lc1 + 1]
        blocker_pixels = int(np.count_nonzero((rect != BG) & (rect != BLUE)))
        if blocker_pixels <= 0 and int(np.count_nonzero(rect == BLUE)) <= largest["area"]:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        zones = _make_zones(h, w)
        if zones is None:
            continue

        red_count, green_count = random.choice([(2, 2), (3, 2), (2, 3), (3, 3)])
        high_color = random.choice([RED, GREEN])
        if high_color == RED:
            red_areas = random.sample(range(8, 13), red_count)
            green_areas = random.sample(range(4, 8), green_count)
        else:
            red_areas = random.sample(range(4, 8), red_count)
            green_areas = random.sample(range(8, 13), green_count)

        groups = {
            RED: _build_group(red_areas, RED),
            GREEN: _build_group(green_areas, GREEN),
        }
        if not _assign_sides(groups):
            continue

        vertical_color = random.choice([RED, GREEN])
        horizontal_color = GREEN if vertical_color == RED else RED
        directions = {
            vertical_color: random.choice(["up", "down"]),
            horizontal_color: random.choice(["left", "right"]),
        }

        zone_map = {color: zone for color, zone in zip(random.sample([RED, GREEN], 2), zones)}
        occupied = np.zeros((h, w), dtype=bool)

        colors_with_three = [color for color, members in groups.items() if len(members) == 3]
        corridor_color = random.choice(colors_with_three) if colors_with_three and random.random() < 0.7 else None

        placed: list[dict] = []
        ok = True
        for color in random.sample([RED, GREEN], 2):
            members = groups[color]
            group_placed = _place_group(
                members,
                zone_map[color],
                occupied,
                directions[color],
                require_corridor=(color == corridor_color),
            )
            if group_placed is None:
                ok = False
                break
            placed.extend(group_placed)
        if not ok:
            continue

        grid = _render_scene((h, w), placed)
        if _validate_scene(grid, placed, directions, corridor_color):
            return grid

    raise RuntimeError("failed to generate a valid ARC input")
